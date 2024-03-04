import numpy as np
import torch
import random
import logging
import time
import pickle
import json
import os
from multiprocessing import Pool

from retro_star.common import args, prepare_starting_molecules, prepare_mlp, \
    prepare_molstar_planner, smiles_to_fp
from retro_star.model import ValueMLP
from retro_star.utils import setup_logger
from mlp_retrosyn.mlp_policies import preprocess
from retro_star.alg.mol_node import MolNode
from retro_star.alg.reaction_node import ReactionNode
from viz_tree.mol_tree_to_paroutes_dict import mol_tree_to_paroutes_dict

from rdkit import RDLogger

# Mute rdkit logger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def retro_plan():
    device = torch.device('cuda' if args.gpu >= 0 else 'cpu')

    starting_mols = prepare_starting_molecules(args.starting_molecules)

    routes = pickle.load(open(args.test_routes, 'rb'))
    logging.info('%d routes extracted from %s loaded' % (len(routes),
                                                         args.test_routes))
    len_list = [len(route) for route in routes]
    max_len = max(len_list)

    one_step = prepare_mlp(args.mlp_templates, args.mlp_model_dump, gpu=args.gpu,
                           realistic_filter=args.realistic_filter)

    # create result folder
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)
    if not os.path.exists(args.result_folder + 'routes_dict/'):
        os.mkdir(args.result_folder + 'routes_dict/')

    if args.use_value_fn:
        model = ValueMLP(
            n_layers=args.n_layers,
            fp_dim=args.fp_dim,
            latent_dim=args.latent_dim,
            dropout_rate=0.1,
            device=device
        ).to(device)
        model_f = '%s/%s' % (args.save_folder, args.value_model)
        logging.info('Loading value nn from %s' % model_f)
        model.load_state_dict(torch.load(model_f,  map_location=device))
        model.eval()

        def value_fn(mol):
            fp = smiles_to_fp(mol, fp_dim=args.fp_dim).reshape(1,-1)
            fp = preprocess(mol, fp_dim=args.fp_dim).reshape(1,-1)
            fp = torch.FloatTensor(fp).to(device)
            v = model(fp).item()
            return v
    else:
        value_fn = lambda x: 0.

    plan_handle = prepare_molstar_planner(
        one_step=one_step,
        value_fn=value_fn,
        starting_mols=starting_mols,
        expansion_topk=args.expansion_topk,
        iterations=args.iterations,
        viz=args.viz,
        viz_dir=args.viz_dir,
        draw_mols=args.draw_mols,
    )
    num_iters = [50, 100, 200, 300, 400, 500]
    result_iter = {
        'succ': {it: [] for it in num_iters},
        'iter': {it: [] for it in num_iters},
        'fail_idx': {it: [] for it in num_iters},
        'route_lens': {it: [] for it in num_iters},
        'avg_lens': {it: 0 for it in num_iters},
        'avg_iter': {it: 0 for it in num_iters},
        'succ_rate': {it: 0 for it in num_iters},
    }
    evaluation_metrics = {
        # efficiency metrics
        'mol_nodes': [],
        'reaction_nodes': [],
        'avg_mol_nodes': 0,
        'avg_reaction_nodes': 0,
        'succ': [],
        'succ_rate': 0,
        # quality metrics (only for successful routes)
        'value_reference': [],
        'avg_value_reference': 0,
        'value': [],
        'avg_value': 0,
        'length': [],
        'avg_length': 0,
    }
    result = {
        'succ': [],
        'cumulated_time': [],
        'iter': [],
        'routes': [],
        'route_costs': [],
        'route_lens': [],
    }
    num_targets = len(routes)
    t0 = time.time()
    for (i, route) in enumerate(routes):

        target_mol = route[0].split('>')[0]
        succ, msg = plan_handle(target_mol, i)

        result['succ'].append(succ)
        result['cumulated_time'].append(time.time() - t0)
        result['iter'].append(msg[1])
        result['routes'].append(msg[0])
        evaluation_metrics['succ'].append(succ)
        if succ:
            # save successful route
            succ_route = mol_tree_to_paroutes_dict(msg[2].root)
            f = open(args.result_folder + 'routes_dict/' + 'mol_{}.json'.format(i+1), 'w')
            json.dump(succ_route, f, indent=4)
            f.close()

            result['route_costs'].append(msg[0].total_cost)
            result['route_lens'].append(msg[0].length)
            evaluation_metrics['value_reference'].append(succ_route["value_reference"])
            evaluation_metrics['value'].append(succ_route["succ_value"])
            evaluation_metrics['length'].append(msg[0].length)
        else:
            result['route_costs'].append(None)
            result['route_lens'].append(None)
            evaluation_metrics['value_reference'].append(0)
            evaluation_metrics['value'].append(0)
            evaluation_metrics['length'].append(0)

        for key in result_iter['succ'].keys():
            if succ and msg[1]<=key:
                result_iter['succ'][key].append(True)
                result_iter['route_lens'][key].append(msg[0].length)
            else:
                result_iter['succ'][key].append(False)
                result_iter['route_lens'][key].append(max_len*2)
                result_iter['fail_idx'][key].append(i+1)
            result_iter['iter'][key].append(min(msg[1] ,key))

        evaluation_metrics['mol_nodes'].append(len(msg[2].mol_nodes))
        evaluation_metrics['reaction_nodes'].append(len(msg[2].reaction_nodes))

        # print logging info
        tot_num = i + 1
        tot_succ = np.array(result['succ']).sum()
        avg_time = (time.time() - t0) * 1.0 / tot_num
        avg_iter = np.array(result['iter'], dtype=float).mean()
        logging.info('Mol: %d | avg mol %d | avg reaction: %d' %
                     (tot_num, len(msg[2].mol_nodes), len(msg[2].reaction_nodes)))
        logging.info('Succ: %d/%d/%d | avg time: %.2f s | avg iter: %.2f' %
                     (tot_succ, tot_num, num_targets, avg_time, avg_iter))

    for key in result_iter['succ'].keys():
        result_iter['avg_lens'][key] = np.sum(result_iter['route_lens'][key]) / len(result_iter['succ'][key])
        result_iter['avg_iter'][key] = np.sum(result_iter['iter'][key]) / len(result_iter['iter'][key])
        result_iter['succ_rate'][key] = np.sum(result_iter['succ'][key]) / len(result_iter['succ'][key])

    evaluation_metrics['avg_mol_nodes'] = np.sum(evaluation_metrics['mol_nodes']) / len(evaluation_metrics['mol_nodes'])
    evaluation_metrics['avg_reaction_nodes'] = np.sum(evaluation_metrics['reaction_nodes']) / len(evaluation_metrics['reaction_nodes'])
    evaluation_metrics['succ_rate'] = np.sum(evaluation_metrics['succ']) / len(evaluation_metrics['succ'])
    evaluation_metrics['avg_value_reference'] = np.sum(evaluation_metrics['value_reference']) / np.sum(evaluation_metrics['succ'])
    evaluation_metrics['avg_value'] = np.sum(evaluation_metrics['value']) / np.sum(evaluation_metrics['succ'])
    evaluation_metrics['avg_length'] = np.sum(evaluation_metrics['length']) / np.sum(evaluation_metrics['succ'])

    result_iter_path = args.result_folder + '/result_iter.json'
    evaluation_metrics_path = args.result_folder + '/evaluation_metrics.json'
    with open(result_iter_path, 'w') as f1, open(evaluation_metrics_path, 'w') as f2:
        json.dump(result_iter, f1, indent=4)
        json.dump(evaluation_metrics, f2, indent=4)

    f = open(args.result_folder + '/plan.pkl', 'wb')
    pickle.dump(result, f)
    f.close()


if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('plan.log')

    retro_plan()
