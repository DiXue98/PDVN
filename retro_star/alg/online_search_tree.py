import numpy as np
from queue import Queue
import logging
import networkx as nx
from graphviz import Digraph
from retro_star.alg.mdtree_mol_node import MDTreeMolNode
from retro_star.alg.mdtree_reaction_node import MDTreeReactionNode

class OnlineSearchTree:
    def __init__(self, target_mol, known_mols, value_fn, prior_fn, args):
        self.target_mol = target_mol
        self.known_mols = known_mols
        self.value_fn = value_fn
        self.prior_fn = prior_fn
        self.args = args
        self.value = 0
        self.depth = 0
        self.mol_nodes = []
        self.reaction_nodes = []
        self.nodes_buffer = Queue()

        self.root = self._add_mol_node(target_mol, None, [])
        self.succ = target_mol in known_mols
        self.simulate_succ = False

        if self.succ:
            logging.info('Synthesis route found: target in starting molecules')

    def _add_mol_node(self, mol, parent, ancestors):
        is_known = mol in self.known_mols

        init_value = 0.0 if is_known else self.value_fn(mol)
        prior = 1.0 if is_known else self.prior_fn(mol)

        mol_node = MDTreeMolNode(
            mol=mol,
            parent=parent,
            is_known=is_known,
            init_value=init_value,
            prior=prior,
        )
        self.depth = max(self.depth, mol_node.depth)

        if mol in ancestors:
            mol_node.repeat = True

        self.mol_nodes.append(mol_node)
        mol_node.id = len(self.mol_nodes)

        return mol_node

    def _add_reaction_and_mol_nodes(self, mols, parent, template, template_idx, ancestors,
                                    score=None, nll=None, nll_reference=None):
        reaction_node = MDTreeReactionNode(parent, template, template_idx, score=score,
                                           nll=nll, nll_reference=nll_reference)
        for mol in mols:
            self._add_mol_node(mol, reaction_node, ancestors)

        self.reaction_nodes.append(reaction_node)
        reaction_node.id = len(self.reaction_nodes)

        return reaction_node

    def mcts_expand(self, mol_node, reactant_lists, templates, templates_idx,
                    scores, nlls=None, nlls_reference=None):
        assert mol_node.open and not mol_node.children

        # deadend case #1 (invalid): cannot pass rdkit
        if templates is None:      # No expansion results
            mol_node.is_deadend = mol_node.is_invalid = True
            logging.info('Expansion fails on %s!' % mol_node.mol)
            return

        ancestors = mol_node.get_ancestors()
        for i in range(len(reactant_lists)):
            # remove repeated molecules
            for mol in reactant_lists[i]:
                if mol in ancestors:
                    break
            else:
                reaction_node = self._add_reaction_and_mol_nodes(reactant_lists[i], mol_node, templates[i],
                                                                 templates_idx[i], ancestors, score=scores[i],
                                                                 nll=nlls[i], nll_reference=nlls_reference[i])

                # initialize reaction node value
                assert reaction_node.children
                reaction_node.Q = reaction_node.cost
                reaction_node.R = 1
                reaction_node.succ = True
                for mol in reaction_node.children:
                    # deadend case #2 (long): depth execeeds the limit
                    if mol.depth >= self.args.depth and not mol.is_known: # TODO: make it a hyper-parameter
                        mol.is_deadend = True
                        mol.open = False

                    reaction_node.Q += mol.init_value
                    reaction_node.R *= mol.prior
                    reaction_node.succ &= mol.succ
                    reaction_node.is_deadend |= mol.is_deadend
                reaction_node.open = False

        # two cases may end up here:
        # deadend case #3 (repeat, w/o children):all reactions lead to molecules appeared in ancestors
        # deadend case #4 (long, w children): all reactions lead to deadends
        mol_node.is_deadend = all(reaction.is_deadend for reaction in mol_node.children)
        mol_node.succ = any(reaction.succ for reaction in mol_node.children)

    def mcts_backup(self, mol_node):
        mol_node.N += 1
        if mol_node.is_deadend:
            if mol_node.is_invalid:
                mol_node.V = 5.0
                mol_node.R = 0.0
            else:
                mol_node.V = mol_node.init_value
                mol_node.R = mol_node.prior
            value = 5.0
            prior = 0.0 # TODO: what if set to 0 ?
        elif mol_node.is_known:
            value = 0.0
            prior = 1.0
        else:
            assert len(mol_node.children)

            if mol_node.succ:
                mol_node.V = min(reaction.Q for reaction in mol_node.children if reaction.succ)
                mol_node.R = 1.0
            else:
                mol_node.V = mol_node.init_value
                mol_node.R = mol_node.prior
            value, prior = mol_node.V,  mol_node.R

            mol_node.L = min(child.L for child in mol_node.children if not child.is_deadend)

        if not mol_node.is_simulation_root:
            mol_node.parent.mcts_backup(value, prior, mol_node, mol_node,
                                        use_single_network=self.args.use_single_network)

    def expand(self, mol_node, reactant_lists, templates, templates_idx):
        assert not mol_node.is_known and not mol_node.children

        if templates is None:      # No expansion results
            mol_node.is_invalid_deadend = True
            return

        # check if the molecule can be solved at next step
        mol_succ, reactions_succ = False, []
        for reactants in reactant_lists:
            assert len(reactants)
            reaction_succ = True
            for reactant in reactants:
                if reactant not in self.known_mols:
                    reaction_succ = False
            reactions_succ.append(reaction_succ)
            mol_succ |= reaction_succ

        # If solved at next step, then remove the siblings of successful reactions
        if mol_succ:
            reactant_lists = [reactant_lists[i] for i, reaction_succ in
                              enumerate(reactions_succ) if reaction_succ]

        assert mol_node.open
        ancestors = mol_node.get_ancestors()
        for i in range(len(reactant_lists)):
            self._add_reaction_and_mol_nodes(reactant_lists[i], mol_node, templates[i],
                                             templates_idx[i], ancestors)

    def backup(self):
        def dfs(mol_node):
            if mol_node.children:
                mol_node.value, mol_node.succ = np.inf, False
                for reaction_node in mol_node.children:
                    reaction_node.value, reaction_node.succ = 0.1, True # reaction cost: 0.1
                    for child in reaction_node.children:
                        dfs(child)
                        reaction_node.value += child.value
                        reaction_node.succ &= child.succ
                    if not reaction_node.is_invalid_template:                        
                        mol_node.value = min(mol_node.value, reaction_node.value)
                    mol_node.succ |= reaction_node.succ
            else:
                if mol_node.succ:                  # successful node
                    mol_node.value = 0.0
                elif mol_node.is_invalid_template: #
                    mol_node.value = 5.0
                elif mol_node.is_invalid_deadend:  # no invalid template
                    mol_node.value = 5.0
                else:                              # too long
                    mol_node.value = 0.0

        dfs(self.root)
        self.value = self.root.value
        self.succ = self.root.succ

    def viz_search_tree(self, viz_file):
        G = Digraph('G', filename=viz_file)
        G.attr(rankdir='LR')
        G.attr('node', shape='box')
        G.format = 'pdf'

        node_queue = Queue()
        node_queue.put((self.root, None))
        while not node_queue.empty():
            node, parent = node_queue.get()

            if node.open:
                color = 'lightgrey'
            else:
                color = 'aquamarine'

            if hasattr(node, 'mol'):
                shape = 'box'
            else:
                if not node.N:
                    continue
                shape = 'rarrow'

            if node.succ:
                color = 'lightblue'
                if hasattr(node, 'mol') and node.is_known:
                    color = 'lightyellow'

            G.node(node.serialize(), shape=shape, color=color, style='filled')

            label = ''
            if hasattr(parent, 'mol'):
                label = 'Q: %.3f, R: %.3f, P: %.3f, N: %.3f, L: %.3f' % (node.Q, node.R or 1, node.P, node.N, node.L or 1e3)
            else:
                label = 'N: %d, R: %.3f, prior: %.3f, init value: %.3f, L: %.3f, value target: %.3f, prior_target: %.3f, temp_target: %.3f' % (node.N, node.R or node.prior, node.prior, node.init_value, node.L or 1e3, node.value_target or 0, node.prior_target or 0, node.template_target or 0)
                if node.children:
                    label += ', V: %.3f' % (node.V or node.init_value)
            if parent is not None:
                G.edge(parent.serialize(), node.serialize(), label=label)

            if node.children is not None:
                for c in node.children:
                    node_queue.put((c, node))

        G.render()
