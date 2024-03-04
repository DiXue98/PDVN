import numpy as np
import logging


class MDTreeMolNode:
    def __init__(self, mol, parent=None, is_known=False, init_value=None, prior=None):
        self.mol = mol
        self.value = 0 # only used in RL
        self.init_value = init_value # initial value given by the cost value function, or 0 if use_value_fn==False
        self.prior = prior # initial value given by the synthesizability value function
        self.parent = parent
        self.cumulative_nll = 0
        self.cumulative_nll_reference = 0

        self.id = -1
        if self.parent is None:
            self.depth = 0
        else:
            self.depth = self.parent.depth

        self.is_known = is_known
        self.children = []
        self.succ = False
        self.open = True    # before expansion: True, after expansion: False
        self.is_invalid_template = False
        self.is_invalid_deadend = False

        # properties reserved for MCTS
        self.V = None # historically averaged value
        self.R = None # historically averaged prior
        self.L = 0.0
        self.N = 0
        self.is_simulation_root = False # mol_node.is_simulation_root == False means mol_node has not been taken as the root node for simulations 
        self.is_taken = False
        self.is_deadend = False
        self.is_invalid = False
        self.simulate_succ = False
        self.succ_len = np.inf # the (minimal) length to building blocks
        self.value_target = np.inf
        self.prior_target = None
        self.template_target = None
        
        if is_known:
            assert self.init_value == 0
            self.open = False
            self.succ = True

        if parent is not None:
            parent.children.append(self)

    def generate_training_dataset(self):
        if len(self.children) == 0:
            return [], [], []
        mols, vals, tmps = [self.mol], [self.value], [self.children[0].template_idx]
        for reaction in self.children:
            m, v, t = reaction.generate_training_dataset()
            mols += m
            vals += v
            tmps += t
        return mols, vals, tmps
    
    def init_mcts_node(self):
        if self.is_known:
            self.V = self.init_value
        else:
            self.succ = False
            for reaction in self.children:
                self.succ |= reaction.succ

            if self.children:
                if self.succ:
                    self.V = min(reaction.Q for reaction in self.children)
                else:
                    self.V = self.init_value
            else:
                self.V = 5 # value for invalid molecules

    def mcts_backup(self, value, prior, from_child, from_expanded_node, use_single_network=False):
        assert not self.is_known

        self.succ |= from_child.succ
        self.is_deadend = all(reaction.is_deadend for reaction in self.children)
        self.is_invalid = all(reaction.is_invalid for reaction in self.children)
        if not from_expanded_node.is_deadend or use_single_network:
            self.V = (self.V * self.N + value) / (self.N + 1)
        self.R = (self.R * self.N + prior) / (self.N + 1)
        self.L = min(child.L for child in self.children)
        self.N += 1

        if not self.is_simulation_root:
            self.parent.mcts_backup(value, prior, self, from_expanded_node, use_single_network=use_single_network)

    def serialize(self):
        text = '%d | %s' % (self.id, self.mol)
        # text = '%d | %s | pred %.2f | value %.2f | target %.2f' % \
        #        (self.id, self.mol, self.pred_value, self.v_self(),
        #         self.v_target())
        return text

    def get_ancestors(self):
        if self.parent is None:
            return {self.mol}

        ancestors = self.parent.parent.get_ancestors()
        ancestors.add(self.mol)
        return ancestors