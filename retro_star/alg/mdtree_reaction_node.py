import numpy as np
import logging


class MDTreeReactionNode:
    def __init__(self, parent, template, template_idx, score=None,
                 nll=None, nll_reference=None):
        self.parent = parent
        
        self.depth = self.parent.depth + 1
        self.id = -1

        self.template = template
        self.template_idx = template_idx
        self.cost = 0.1 # TODO: add to hyper-parameter, or config
        self.score = score
        self.children = []
        self.value = 0
        self.succ = None    # successfully found a valid synthesis route
        self.open = True    # before expansion: True, after expansion: False
        self.is_invalid_template = False
        self.nll = nll # negative log likelihood given by the trained model
        self.nll_reference = nll_reference # negative log likelihood given by the reference model
        self.cumulative_nll = 0
        self.cumulative_nll_reference = 0
        parent.children.append(self)

        # properties reserved for MCTS
        self.Q = 0
        self.P = self.score
        self.L = nll
        self.N = 0
        self.is_taken = False
        self.is_deadend = False
        self.is_invalid = False
        self.simulate_succ = False
        self.succ_len = np.inf # the (minimal) length to building blocks
        self.value_target = np.inf # the (minimal) cost to realize the reaction

    def init_mcts_node(self):
        assert self.open

        self.Q = 0
        self.P = self.score
        self.N = 0

        self.succ = True
        self.Q = self.cost
        for mol in self.children:
            self.Q += mol.init_value
            self.succ &= mol.succ

        self.open = False

    def mcts_backup(self, value, prior, from_child, from_expanded_node, use_single_network=False):
        self.Q = new_Q = self.cost
        self.R = new_R = 1.0
        self.succ = True
        self.is_deadend = self.is_invalid = False

        for mol in self.children:
            self.succ &= mol.succ
            self.is_deadend |= mol.is_deadend
            self.is_invalid |= mol.is_invalid
            mol_V = mol.V or mol.init_value # building block molecules do not have the attributes V
            mol_R = mol.R or mol.prior # building block molecules do not have the attributes V
            self.Q += mol_V
            self.R *= mol_R
            new_Q += value if mol is from_child else mol_V
            new_R *= prior if mol is from_child else mol_R
        self.N += 1
        self.L = self.nll + sum(child.L for child in self.children)

        self.parent.mcts_backup(new_Q, new_R, self, from_expanded_node, use_single_network=use_single_network)

    def generate_training_dataset(self):
        mols, vals, tmps = [], [], []
        for mol in self.children:
            m, v, t = mol.generate_training_dataset()
            mols += m
            vals += v
            tmps += t
        return mols, vals, tmps

    def propagate(self, v_delta, exclude=None):
        if exclude is None:
            self.target_value += v_delta

        for child in self.children:
            if exclude is None or child.mol != exclude:
                for grandchild in child.children:
                    grandchild.propagate(v_delta)

    def serialize(self):
        return '%d' % (self.id)
        # return '%d | value %.2f | target %.2f' % \
        #        (self.id, self.v_self(), self.v_target())