import os
import numpy as np
import logging
from queue import Queue
from retro_star.alg.online_search_tree import OnlineSearchTree

def online_mcts(target_mol, target_mol_id, starting_mols, expand_fn, value_fn, prior_fn,
                iterations, args, viz=False, viz_dir=None, test=False):
    mol_tree = OnlineSearchTree(
        target_mol=target_mol,
        known_mols=starting_mols,
        value_fn=value_fn,
        prior_fn=prior_fn,
        args=args,
    )
    result = {
        "tot_model_calls": 0,
        "num_succ_before_take": 0,
        "num_take_unsucc_when_succ": 0,
        "num_deadends": 0,
        "num_building_blocks": 0,
        "num_unexpanded": 0,
        "mols_r": [], "pris": [], "mols_v": [], "vals": [], "mols_t": [], "tmps": [],
    }
    mol_tree.nodes_buffer.put(mol_tree.root)
    if not mol_tree.succ:
        while (not mol_tree.nodes_buffer.empty()
               and result["tot_model_calls"] < iterations): # TODO: change the name of iterations
            m_next = mol_tree.nodes_buffer.get()

            # simulation
            # TODO: restrict to 500 calls
            available_num_model_calls = min(args.num_simulations, iterations - result["tot_model_calls"])
            num_model_calls = simulate(mol_tree, m_next, available_num_model_calls, expand_fn, args, test=test)
            # assert num_model_calls <= available_num_model_calls
            result["tot_model_calls"] += num_model_calls

            # fail condition: m_next is a deadend after simulation
            if m_next.is_deadend:
                break

            # choose reaction
            assert len(m_next.children)
            reaction = take_reaction(mol_tree, m_next, test=test)
            if m_next.succ:
                result["num_succ_before_take"] += 1
                if not reaction.succ:
                    result["num_take_unsucc_when_succ"] += 1
        else:
            if mol_tree.nodes_buffer.empty():
                mol_tree.succ = True

        final_backup(mol_tree, result, args)
        # assert mol_tree.succ == mol_tree.simulate_succ, f"id {target_mol_id}, succ: {mol_tree.succ}, simulate_succ: {mol_tree.simulate_succ}"
        if not test:
            (
                result["mols_r"], result["pris"],
                result["mols_v"], result["vals"],
                result["mols_t"], result["tmps"],
            ) = generate_training_data(mol_tree, args)

    if viz:
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        f = '%s/mol_%d_search_tree' % (viz_dir, target_mol_id)
        mol_tree.viz_search_tree(f)

    return mol_tree.succ, (mol_tree, result["tot_model_calls"]), result # TODO: simplify this

def simulate(mol_tree, root_mol, available_num_model_calls, expand_fn, args, test=False):
    assert not (test and root_mol.succ)
    assert not root_mol.is_deadend

    root_mol.is_simulation_root = True
    model_calls_cnt = 0

    while root_mol.N < args.num_simulations:
        if root_mol.is_deadend:
            return model_calls_cnt

        mol_node = root_mol
        # select a molecule to expand
        C = args.PUCT_coef
        while mol_node.children:
            assert not mol_node.is_deadend
            assert not all(reaction.is_deadend for reaction in mol_node.children)
            # select reaction node
            Q = np.array([reaction_node.Q for reaction_node in mol_node.children])

            P = np.array([reaction_node.P for reaction_node in mol_node.children])
            L = np.array([reaction_node.L for reaction_node in mol_node.children])
            # P = np.exp(-L)
            P /= P.sum()

            N = np.array([reaction_node.N for reaction_node in mol_node.children])

            R = np.array([reaction_node.R for reaction_node in mol_node.children])
            assert np.logical_and(0.0 <= R, R <= 1.0).all()
            Q = Q.clip(0, 5) # TODO: to be tuned
            # U = (1 - R) * (-np.log(R)) + R * Q
            U = R * Q + (1 - R) * 5.0
            # U = np.exp(-U)
            # U = R

            # inject exploration noise to the root node
            # if not test and mol_node is root_mol:
            # # if not test:
            # # if mol_node is root_mol:
            #     dirichlet_alpha = 0.25
            #     exploration_nose_faction = 0.25
            #     noise = np.random.dirichlet([dirichlet_alpha] * len(P))
            #     P = P * (1 - exploration_nose_faction) + noise * exploration_nose_faction

            # calculate PUCT
            if args.use_single_network:
                PUCT = - Q + C * P * np.sqrt(np.sum(N)) / (1 + N) # single value
            else:
                PUCT = - U + C * P * np.sqrt(np.sum(N)) / (1 + N) # dual value
                # PUCT = - U + C * P * np.sqrt(np.sum(N)) / (1e-5 + N) # dual value

            # mask deadends
            is_deadend = np.array([reaction.is_deadend for reaction in mol_node.children])
            assert not all(reaction.is_deadend for reaction in mol_node.children), len(mol_node.children)
            PUCT[is_deadend] = -np.inf

            # select reaction node
            reaction_node = mol_node.children[PUCT.argmax()]

            # select molecule node
            not_expanded = np.array([mol.open for mol in reaction_node.children])
            not_succ = np.array([not mol.succ for mol in reaction_node.children])
            not_expanded_and_not_succ = not_expanded & not_succ
            
            if not_expanded_and_not_succ.any():
                # randomly select an unexpanded and unsuccessful molecule node
                next_idx = np.random.choice(len(reaction_node.children), p=not_expanded_and_not_succ/not_expanded_and_not_succ.sum())
            elif not_succ.any():
                # randomly select an unsuccessful molecule node
                next_idx = np.random.choice(len(reaction_node.children), p=not_succ/not_succ.sum())
            else:
                # reaction_node is successful, then select a child randomly
                assert root_mol.succ
                next_idx = np.random.choice(len(reaction_node.children))

            mol_node = reaction_node.children[next_idx]
            assert not mol_node.is_deadend

        # mol_node is either an (1) unexpanded node or a (2) building block
        if mol_node.open:
            # expand
            result = expand_fn(mol_node)

            if result is not None and (len(result["template"]) > 0):
                reactants = result['reactants']     
                if 'templates' in result.keys():
                    templates = result['templates']
                else:
                    templates = result['template']
                templates_idx = result['templates_idx']
                scores = result['scores']
                scores_reference = result['scores_reference']
                nlls = 0.0 - np.log(np.clip(np.array(scores), 1e-3, 1.0))
                nlls_reference = 0.0 - np.log(np.clip(np.array(scores_reference), 1e-3, 1.0))

                reactant_lists = []
                for j in range(len(scores)):
                    reactant_list = list(set(reactants[j].split('.')))
                    reactant_lists.append(reactant_list)

                try:
                    mol_tree.mcts_expand(mol_node, reactant_lists, templates, templates_idx,
                                         scores=scores, nlls=nlls, nlls_reference=nlls_reference)
                    model_calls_cnt += 1
                except RecursionError: # Bad target molecule input,
                    # RecursionError: maximum recursion depth exceeded while calling a Python object
                    return model_calls_cnt

            else: # deadends
                mol_tree.mcts_expand(mol_node, None, None, None, None)

        mol_tree.mcts_backup(mol_node)
        mol_node.open = False
        if test and root_mol.succ:
            return model_calls_cnt

    return model_calls_cnt

def take_reaction(mol_tree, mol_node, temperature=1, test=False):
    N = np.array([reaction.N if not reaction.is_deadend else 0 for reaction in mol_node.children])
    # N = np.array([reaction.N + 1 if not reaction.is_deadend else 0 for reaction in mol_node.children])
    if test or temperature == 0:
        template_idx = N.argmax()
    else:
        N_distribution = N ** (1 / temperature)
        N_distribution /= N_distribution.sum()
        try:
            template_idx = np.random.choice(len(N_distribution), p=N_distribution)
        except ValueError:
            N = np.array([1 if not reaction.is_deadend else 0 for reaction in mol_node.children])
            N_distribution = N / N.sum()
            template_idx = np.random.choice(len(N_distribution), p=N_distribution)

    reaction = mol_node.children[template_idx]
    if not (test and mol_node.succ): # during test, if mol_node is successful, then no need to add children to the node_buffer
        for mol in reaction.children:
            if not mol.is_known and not (test and mol.succ):
                mol_tree.nodes_buffer.put(mol)
    mol_node.is_taken = True
    reaction.is_taken = True
    return reaction

def final_backup(mol_tree, result, args):
    '''
    Things to backup: 
        1. simulate succ
        2. trajectory value
        3. template to imitate
    '''

    def dfs(mol_node):
        if mol_node.children:
            for reaction in mol_node.children:
                reaction.simulate_succ = True
                for mol in reaction.children:
                    dfs(mol)
                    reaction.simulate_succ &= mol.simulate_succ
                mol_node.simulate_succ |= reaction.simulate_succ

                if reaction.simulate_succ:
                    reaction.succ_len = 1
                    reaction.value_target = reaction.cost
                    reaction.cumulative_nll = reaction.nll
                    reaction.cumulative_nll_reference = reaction.nll_reference
                    for mol in reaction.children:
                        reaction.succ_len += mol.succ_len
                        reaction.value_target += mol.value_target
                        reaction.cumulative_nll += mol.cumulative_nll
                        reaction.cumulative_nll_reference += mol.cumulative_nll_reference
                        assert (reaction.succ_len is not np.inf and
                                reaction.value_target is not np.inf)
                    
                    if reaction.value_target + 1e-8 < mol_node.value_target: # use 1e-8 to offset the float error
                        assert reaction.succ_len < mol_node.succ_len
                        mol_node.value_target = reaction.value_target
                        mol_node.succ_len = reaction.succ_len
                        mol_node.template_target = reaction.template_idx
                        mol_node.cumulative_nll = reaction.cumulative_nll
                        mol_node.cumulative_nll_reference = reaction.cumulative_nll_reference

            # set value targets when use only cost network
            if not mol_node.simulate_succ and args.use_single_network:
                mol_node.value_target = mol_node.V
                mol_node.value_target += 0.2 if mol_node.is_deadend else 0

            # add penalty to the prior
            if mol_node.is_invalid:
                mol_node.prior_target = 0.0
            elif mol_node.is_deadend:
                mol_node.prior_target = mol_node.R * 0.8
            else:
                mol_node.prior_target = 1.0 if mol_node.simulate_succ else mol_node.R * 0.8
                # mol_node.prior_target = 1.0 if mol_node.simulate_succ else mol_node.R

        else:
            if mol_node.is_known:
                mol_node.simulate_succ = True
                mol_node.succ_len = 0
                mol_node.value_target = 0.0
                mol_node.prior_target = 1.0
            elif mol_node.is_invalid:
                mol_node.value_target = 5.0
                mol_node.prior_target = 0.0
            else:
                # unexpanded
                assert not mol_node.simulate_succ
                mol_node.value_target = mol_node.init_value
                mol_node.prior_target = mol_node.R or mol_node.prior
                if mol_node.is_deadend:
                    mol_node.value_target += 0.2
                    mol_node.prior_target *= 0.8

            # log into result
            if mol_node.is_known:
                result["num_building_blocks"] += 1
            elif mol_node.is_deadend:
                result["num_deadends"] += 1
            else:
                result["num_unexpanded"] += 1

    dfs(mol_tree.root)
    mol_tree.simulate_succ = mol_tree.root.simulate_succ

def generate_training_data(mol_tree, args):
    mol_queue = Queue()
    mols_r, pris, mols_v, vals, mols_t, tmps = [], [], [], [], [], []
    mol_queue.put(mol_tree.root)
    while not mol_queue.empty():
        node = mol_queue.get()
        if hasattr(node, 'mol'):
            mols_r.append(node.mol)
            pris.append(node.prior_target)
            if node.simulate_succ:
                mols_v.append(node.mol)
                mols_t.append(node.mol)
                vals.append(node.value_target)
                tmps.append(node.template_target)
            elif args.use_single_network:
                mols_v.append(node.mol)
                vals.append(node.value_target)
        for child in node.children:
            if child.children:
                mol_queue.put(child)
            elif hasattr(child, 'mol') and child.is_deadend: # append deadends' targets
                mols_r.append(child.mol)
                pris.append(child.prior_target)
                if args.use_single_network:                    
                    mols_v.append(child.mol)
                    vals.append(child.value_target)
    return mols_r, pris, mols_v, vals, mols_t, tmps