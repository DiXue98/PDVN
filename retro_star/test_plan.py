import random
import logging
import time
import pickle
import json
import os
import pathlib
from os.path import dirname, abspath

from retro_star.common import args, prepare_mlp
from retro_star.model import ValueMLP, PriorMLP
from retro_star.utils import setup_logger
from retro_star.runner import SerialRunner, ParallelRunner
from rdkit import Chem

import numpy as np
import torch
from rdkit import RDLogger

# Mute rdkit logger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def test_plan():
    device = torch.device('cpu') # use cpu in the main process, but gpu in children processes

    # test routes
    routes = pickle.load(open(args.test_routes, 'rb'))
    logging.info('%d routes extracted from %s loaded' % (len(routes),
                                                         args.train_routes))

    target_mols = [Chem.CanonSmiles(route[0].split('>')[0]) for route in routes]
    num_targets = len(target_mols)
    target_mols_id = list(range(num_targets))
    max_len = 16 # for USPTO test molecules

    one_step = prepare_mlp(args.mlp_templates, args.mlp_model_dump,
                           realistic_filter=args.realistic_filter)

    # prepare value function
    value_net = ValueMLP(
        n_layers=args.n_layers,
        fp_dim=args.fp_dim,
        latent_dim=args.latent_dim,
        dropout_rate=0.1,
        device=device   
    ).to(device)
    value_net.eval()

    # prepare prior function
    prior_net = PriorMLP(
        n_layers=args.n_layers,
        fp_dim=args.fp_dim,
        latent_dim=args.latent_dim,
        dropout_rate=0.1,
        device=device   
    ).to(device)
    prior_net.eval()

    if args.runner == 'serial':
        runner = SerialRunner(args)
    elif args.runner == 'parallel':
        runner = ParallelRunner(args)
    else:
        raise NotImplementedError

    result_folder = pathlib.Path(args.result_folder)
    assert result_folder.exists()
    for checkpoint_folder in result_folder.iterdir():
        if not checkpoint_folder.is_dir():
            continue

        t0 = time.time()
        num_iters = [50, 100, 200, 300, 400, 500]
        result_iter = {
            'succ': {it: [None] * num_targets for it in num_iters},
            'iter': {it: [None] * num_targets for it in num_iters},
            'fail_idx': {it: [] for it in num_iters},
            'route_lens': {it: [None] * num_targets for it in num_iters},
            'avg_lens': {it: 0 for it in num_iters},
            'avg_iter': {it: 0 for it in num_iters},
            'succ_rate': {it: 0 for it in num_iters},
        }
        evaluation_metrics = {
            # efficiency metrics
            'mol_nodes': [None] * num_targets,
            'reaction_nodes': [None] * num_targets,
            'avg_mol_nodes': 0,
            'avg_reaction_nodes': 0,
            'succ': [None] * num_targets,
            'succ_rate': 0,
            # quality metrics (only for successful routes)
            'value_reference': [0] * num_targets,
            'avg_value_reference': 0,
            'value': [0] * num_targets,
            'avg_value': 0,
            'length': [0] * num_targets,
            'avg_length': 0,
        }

        one_step_ckpt = checkpoint_folder.joinpath("one_step", "rollout_model.ckpt")
        value_ckpt = checkpoint_folder.joinpath("one_step", "value_fn.ckpt")
        prior_ckpt = checkpoint_folder.joinpath("one_step", "prior_fn.ckpt")

        # load models
        one_step.net.load_state_dict(torch.load(one_step_ckpt,  map_location=device))
        try:
            value_net.load_state_dict(torch.load(value_ckpt,  map_location=device))
        except:
            print(f"value network cannot be found at {value_ckpt}")
        try:
            prior_net.load_state_dict(torch.load(prior_ckpt,  map_location=device))
        except:
            print(f"Prior network cannot be found at {prior_ckpt}")

        load_model_to_runner(one_step.net, value_net, prior_net, runner, args.n_gpus)
        if args.method == "mcts":
            _, _, _, _, _, _, plan_results = runner.mcts(target_mols, target_mols_id, test=True)
        elif args.method == "retro":
            plan_results = runner.retro(target_mols, target_mols_id)
        else:
            raise NotImplementedError

        # log results
        for i, id in enumerate(plan_results["ids"]):
            for key in num_iters:
                if plan_results["succ"][i] and plan_results["iter"][i] <= key:
                    result_iter["succ"][key][id] = True
                    if args.method == "mcts":
                        result_iter['route_lens'][key][id] = plan_results["route_lens"][i]
                else:
                    result_iter['succ'][key][id] = False                  
                    result_iter['fail_idx'][key].append(id)  
                    if args.method == "mcts":
                        result_iter['route_lens'][key][id] = max_len*2
                result_iter['iter'][key][id] = min(plan_results["iter"][i], key)

            evaluation_metrics["succ"][id] = plan_results["succ"][i]
            evaluation_metrics["mol_nodes"][id] = plan_results["mol_nodes"][i]
            evaluation_metrics["reaction_nodes"][id] = plan_results["reaction_nodes"][i]
            if plan_results["succ"][i]:
                evaluation_metrics["value_reference"][id] = plan_results["nll_reference"][i]
                evaluation_metrics["value"][id] = plan_results["nll"][i]
                evaluation_metrics["length"][id] = plan_results["succ_len"][i]

        # calculate average values
        for key  in num_iters:
            result_iter["avg_iter"][key] = np.array(result_iter["iter"][key]).mean().item()
            result_iter["succ_rate"][key] = np.array(result_iter["succ"][key]).mean().item()
            if args.method == "mcts":
                result_iter["avg_lens"][key] = np.array(result_iter["route_lens"][key]).mean().item()
        evaluation_metrics["avg_mol_nodes"] = np.array(evaluation_metrics["mol_nodes"]).mean().item()
        evaluation_metrics["avg_reaction_nodes"] = np.array(evaluation_metrics["reaction_nodes"]).mean().item()
        evaluation_metrics["succ_rate"] = np.array(evaluation_metrics["succ"]).mean().item()
        evaluation_metrics["avg_value_reference"] = np.array(evaluation_metrics["value_reference"])[evaluation_metrics["succ"]].mean().item()
        evaluation_metrics["avg_value"] = np.array(evaluation_metrics["value"])[evaluation_metrics["succ"]].mean().item()
        evaluation_metrics["avg_length"] = np.array(evaluation_metrics["length"])[evaluation_metrics["succ"]].mean().item()

        avg_time = (time.time() - t0) / num_targets
        avg_iter = np.array(plan_results['iter']).mean().item()
        dataset_name = os.path.basename(args.test_routes).split('.')[0]
        method = args.method
        logging.info('Finish testing %s with checkpoint %s on dataset %s'%
                     (args.method, checkpoint_folder.name, os.path.basename(dataset_name)))
        logging.info('Succ: %d/%d | avg time: %.2f s | avg iter: %.2f'%
            (sum(evaluation_metrics["succ"]), num_targets, avg_time, avg_iter))   
        if args.method == "mcts":
            avg_depth = np.array(plan_results['route_lens']).mean().item()
            logging.info('Shortest route cost: %.2f | avg depth: %.2f' % 
                        (np.array(plan_results['route_shortest_cost']).mean().item(), avg_depth))
            logging.info('Unsuccessful reactions/Successful molecules: %d/%d' %
                        (plan_results['num_take_unsucc_when_succ'], plan_results['num_succ_before_take']))

        if args.use_value_fn == True:
            result_iter_path = checkpoint_folder.joinpath(f"{method}_wvalue_{dataset_name}", "result_iter.json")
            evaluation_metrics_path = checkpoint_folder.joinpath(f"{method}_wvalue_{dataset_name}", "evaluation_metrics.json")
        else:
            result_iter_path = checkpoint_folder.joinpath(f"{method}_{dataset_name}", "result_iter.json")
            evaluation_metrics_path = checkpoint_folder.joinpath(f"{method}_{dataset_name}", "evaluation_metrics.json")
        if not result_iter_path.parent.exists():
            result_iter_path.parent.mkdir()
        with open(result_iter_path, 'w') as f1, open(evaluation_metrics_path, 'w') as f2:
            json.dump(result_iter, f1, indent=4)
            json.dump(evaluation_metrics, f2, indent=4)

    runner.close()
    logging.info('The runner closes gracefully')

def load_model_to_runner(model_net, value_net, prior_net, runner, n_gpus=1):
    runner.load_model(model_net.state_dict(), prior_net.state_dict(), value_net.state_dict())

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('plan.log')

    test_plan()
