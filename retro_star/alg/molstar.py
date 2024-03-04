import os
import numpy as np
import logging
from retro_star.alg.mol_tree import MolTree


def molstar(target_mol, target_mol_id, starting_mols, expand_fn, value_fn,
            iterations, viz=False, viz_dir=None, draw_mols=True):
    mol_tree = MolTree(
        target_mol=target_mol,
        known_mols=starting_mols,
        value_fn=value_fn
    )

    i = -1

    if not mol_tree.succ:
        for i in range(iterations):
            scores = []
            for m in mol_tree.mol_nodes:
                if m.open:
                    scores.append(m.v_target())
                else:
                    scores.append(np.inf)
            scores = np.array(scores)

            if np.min(scores) == np.inf:
                logging.info('No open nodes!')
                break

            metric = scores

            mol_tree.search_status = np.min(metric)
            m_next = mol_tree.mol_nodes[np.argmin(metric)]
            assert m_next.open

            result = expand_fn(m_next.mol)

            if result is not None and (len(result['scores']) > 0):
                reactants = result['reactants']
                scores = result['scores']
                scores_reference = result['scores_reference']
                costs = 0.0 - np.log(np.clip(np.array(scores), 1e-3, 1.0))
                costs_reference = 0.0 - np.log(np.clip(np.array(scores_reference), 1e-3, 1.0))
                # costs = 1.0 - np.array(scores)
                if 'templates' in result.keys():
                    templates = result['templates']
                else:
                    templates = result['template']

                reactant_lists = []
                for j in range(len(scores)):
                    reactant_list = list(set(reactants[j].split('.')))
                    reactant_lists.append(reactant_list)

                assert m_next.open
                try:
                    # succ = mol_tree.expand(m_next, reactant_lists, costs, templates)
                    succ = mol_tree.expand(m_next, reactant_lists, costs, templates, costs_reference=costs_reference)
                except RecursionError: # Bad target molecule input,
                    # RecursionError: maximum recursion depth exceeded while calling a Python object
                    succ = False
                    break

                if succ:
                    break

                # found optimal route
                if mol_tree.root.succ_value <= mol_tree.search_status:
                    break

            else:
                mol_tree.expand(m_next, None, None, None)
                logging.info('Expansion fails on %s!' % m_next.mol)

        logging.info('Final search status | success value | iter: %s | %s | %d'
                     % (str(mol_tree.search_status), str(mol_tree.root.succ_value), i+1))

    best_route = None
    if mol_tree.succ:
        best_route = mol_tree.get_best_route()
        assert best_route is not None

    if viz:
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        if mol_tree.succ:
            if best_route.optimal:
                f = '%s/mol_%d_route_optimal' % (viz_dir, target_mol_id)
            else:
                f = '%s/mol_%d_route' % (viz_dir, target_mol_id)
            if draw_mols:
                mol_tree.viz_tree_with_pic(f)

        f = '%s/mol_%d_search_tree' % (viz_dir, target_mol_id)
        mol_tree.viz_search_tree(f)

    return mol_tree.succ, (best_route, i+1, mol_tree)


def molstar_ens(target_mol, target_mol_id, starting_mols, expand_fn_A, expand_fn_B, value_fn,
            iterations, viz=False, viz_dir=None):
    mol_tree = MolTree(
        target_mol=target_mol,
        known_mols=starting_mols,
        value_fn=value_fn
    )

    i = -1

    if not mol_tree.succ:
        for i in range(iterations):
            scores = []
            for m in mol_tree.mol_nodes:
                if m.open:
                    scores.append(m.v_target())
                else:
                    scores.append(np.inf)
            scores = np.array(scores)

            if np.min(scores) == np.inf:
                logging.info('No open nodes!')
                break

            metric = scores

            mol_tree.search_status = np.min(metric)
            m_next = mol_tree.mol_nodes[np.argmin(metric)]
            assert m_next.open

            result_A = expand_fn_A(m_next.mol)
            result_B = expand_fn_B(m_next.mol)

            if result_A is not None and (len(result_A['scores']) > 0) \
                    and result_B is not None and (len(result_B['scores']) > 0):
                reactants_A = result_A['reactants']
                scores_A = result_A['scores']
                # costs_A = 0.0 - np.log(np.clip(np.array(scores_A), 1e-3, 1.0))
                if 'templates' in result_A.keys():
                    templates_A = result_A['templates']
                else:
                    templates_A = result_A['template']

                reactants_B = result_B['reactants']
                scores_B = result_B['scores']
                # costs_B = 0.0 - np.log(np.clip(np.array(scores_B), 1e-3, 1.0))
                if 'templates' in result_B.keys():
                    templates_B = result_B['templates']
                else:
                    templates_B = result_B['template']

                reactants_AB = reactants_A
                scores_AB = scores_A
                templates_AB = templates_A

                for react_B, score_B, tpl_B in zip(reactants_B, scores_B, templates_B):
                    try:
                        idx = reactants_AB.index(react_B)
                        # if templates_AB[idx] != tpl_B:
                        #    raise AssertionError
                        scores_AB[idx] += score_B

                    except ValueError:
                        reactants_AB.append(react_B)
                        scores_AB.append(score_B)
                        templates_AB.append(templates_B)

                for idx in range(len(scores_AB)):
                    scores_AB[idx] = scores_AB[idx] * 0.5

                reactants_AB = [x for _, x in sorted(zip(scores_AB, reactants_AB), reverse=True, key=lambda pair: pair[0])]
                templates_AB = [x for _, x in sorted(zip(scores_AB, templates_AB), reverse=True, key=lambda pair: pair[0])]
                scores_AB = sorted(scores_AB, reverse=True)
                costs_AB = 0.0 - np.log(np.clip(np.array(scores_AB), 1e-3, 1.0))

                reactant_lists = []
                for j in range(len(scores_AB)):
                    reactant_list = list(set(reactants_AB[j].split('.')))
                    reactant_lists.append(reactant_list)

                assert m_next.open
                succ = mol_tree.expand(m_next, reactant_lists, costs_AB, templates_AB)

                if succ:
                    break

                # found optimal route
                if mol_tree.root.succ_value <= mol_tree.search_status:
                    break

            else:
                mol_tree.expand(m_next, None, None, None)
                logging.info('Expansion fails on %s!' % m_next.mol)

        logging.info('Final search status | success value | iter: %s | %s | %d'
                     % (str(mol_tree.search_status), str(mol_tree.root.succ_value), i+1))

    best_route = None
    if mol_tree.succ:
        best_route = mol_tree.get_best_route()
        assert best_route is not None

    if viz:
        if not os.path.exists(viz_dir):
            os.makedirs(viz_dir)

        if mol_tree.succ:
            if best_route.optimal:
                f = '%s/mol_%d_route_optimal' % (viz_dir, target_mol_id)
            else:
                f = '%s/mol_%d_route' % (viz_dir, target_mol_id)
            best_route.viz_route(f)

        f = '%s/mol_%d_search_tree' % (viz_dir, target_mol_id)
        mol_tree.viz_search_tree(f)

    return mol_tree.succ, (best_route, i+1)
