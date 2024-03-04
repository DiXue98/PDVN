import random
import logging
import time
import pickle
import json
import os
import datetime
from os.path import dirname, abspath
import subprocess

from retro_star.common import args, prepare_mlp
from retro_star.model import ValueMLP, PriorMLP
from retro_star.utils import setup_logger
from retro_star.pdvn_trainer import PDVNTrainer
from retro_star.runner import SerialRunner, ParallelRunner

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from rdkit import Chem
from rdkit import RDLogger

# Mute rdkit logger
lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)

def retro_mcts():
    # setup tensorboard
    unique_token = '{}__{}'.format(args.name, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    log_dir = os.path.join(dirname(abspath(__file__)), 'runs', unique_token)
    WRITER = SummaryWriter(log_dir=log_dir)

    device = torch.device('cuda' if args.gpu >= 0 else 'cpu')

    # train routes
    routes = pickle.load(open(args.train_routes, 'rb'))
    logging.info('%d routes extracted from %s loaded' % (len(routes),
                                                         args.train_routes))

    one_step = prepare_mlp(args.mlp_templates, args.mlp_model_dump,
                           gpu=args.gpu, realistic_filter=args.realistic_filter)

    # create result folder
    if not os.path.exists(args.result_folder):
        os.makedirs(args.result_folder)

    # add commit hash
    cmd = "git rev-parse --short HEAD"
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    args.commit_hash = output.decode("utf-8").strip()

    # save arguments
    with open(args.result_folder + "args.json", "w") as f:
        json.dump(args.__dict__, f, indent=4)

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

    trainer = PDVNTrainer(args, prior_net, value_net, one_step, device)

    if args.runner == 'serial':
        runner = SerialRunner(args)
    elif args.runner == 'parallel':
        runner = ParallelRunner(args)
    else:
        raise NotImplementedError
    load_model_to_runner(one_step.net, value_net, prior_net, runner, args.n_gpus)

    target_mols = [Chem.CanonSmiles(route[0].split('>')[0]) for route in routes]
    num_targets = len(target_mols)
    train_target_mols_idx = list(range(num_targets))
    tot_num = 0
    last_test_num, last_save_num = 0, 0
    global_result = {}
    timer = {
        'num_mols': [],
        'rollout': [],
        'training': [],
    }
    for epoch in range(args.n_epochs):
        epoch_result = {
            'succ': [False]*num_targets,
            'simulate_succ': [False]*num_targets,
            'simulate_fail_idx': [],
            'iter': [0]*num_targets,
            'route_lens': [0]*num_targets,
            "num_succ_before_take": 0,
            "num_take_unsucc_when_succ": 0,
            "num_deadends": 0,
            "num_building_blocks": 0,
            "num_unexpanded": 0,
        }
        t0 = time.time()
        cur_num = 0
        mols_batch, vals_batch, tmps_batch = [], [], []
        random.shuffle(train_target_mols_idx)
        for i in range(0, num_targets, args.batch_size):
            num_mols = min(args.batch_size, num_targets-cur_num)
            target_mols_id_batch = train_target_mols_idx[i: i+args.batch_size]
            target_mols_batch = [target_mols[idx] for idx in target_mols_id_batch]
            fail_mols = []
            fail_mols_idx = []
            
            # batch plan
            rollout_start = time.time()
            mols_r_batch, pris_batch, mols_v_batch, vals_batch, mols_t_batch, tmps_batch, batch_result = runner.mcts(target_mols_batch, target_mols_id_batch)
            rollout_end = time.time()
            timer['rollout'].append(rollout_end - rollout_start)

            tot_num += num_mols
            cur_num += num_mols
            timer['num_mols'].append(tot_num)
            for i, mol_id in enumerate(batch_result['ids']):
                epoch_result['succ'][mol_id] = batch_result['succ'][i]
                epoch_result['simulate_succ'][mol_id] = batch_result['simulate_succ'][i]
                epoch_result['iter'][mol_id] = batch_result['iter'][i]
                epoch_result['route_lens'][mol_id] = batch_result['route_lens'][i]
                if not epoch_result['succ'][mol_id]:
                    fail_mols.append(target_mols[mol_id])
                    fail_mols_idx.append(mol_id)
                if not epoch_result['simulate_succ'][mol_id]:
                    epoch_result["simulate_fail_idx"].append(mol_id)
            epoch_result["num_succ_before_take"] += batch_result["num_succ_before_take"]
            epoch_result["num_take_unsucc_when_succ"] += batch_result["num_take_unsucc_when_succ"]
            epoch_result["num_deadends"] += batch_result["num_deadends"]
            epoch_result["num_building_blocks"] += batch_result["num_building_blocks"]
            epoch_result["num_unexpanded"] += batch_result["num_unexpanded"]

            training_start = time.time()
            prior_loss = trainer.fit_prior(mols_r_batch, pris_batch, epoch_num=8)
            critic_loss = trainer.fit_value(mols_v_batch, vals_batch, epoch_num=8)
            actor_loss = trainer.fit_model(mols_t_batch, tmps_batch, epoch_num=8)
            training_end = time.time()
            timer['training'].append(training_end - training_start)

            load_model_to_runner(one_step.net, value_net, prior_net, runner, args.n_gpus)

            # testing
            if tot_num - last_test_num >= args.test_interval:
                test_target_mols_idx = np.random.choice(num_targets, size=args.test_batch_size, replace=False)
                test_target_mols_batch = [Chem.CanonSmiles(routes[idx][0].split('>')[0]) for idx in test_target_mols_idx]
                _, _, _, _, _, _, test_batch_result = runner.mcts(test_target_mols_batch, test_target_mols_idx, test=True)
                WRITER.add_scalar('testing/success rate', np.array(test_batch_result['succ']).sum()/len(test_batch_result['succ']), tot_num)
                WRITER.add_scalar('testing/depth', np.array(test_batch_result['route_lens']).sum()/len(test_batch_result['route_lens']), tot_num)
                WRITER.add_scalar('testing/model calls', np.array(test_batch_result['iter']).sum()/len(test_batch_result['iter']), tot_num)
                WRITER.add_scalar('testing/route_shortest_cost', np.array(test_batch_result['route_shortest_cost']).sum()/len(test_batch_result['route_shortest_cost']), tot_num)

                last_test_num = tot_num

            # logging
            tot_succ = np.array(epoch_result['succ']).sum()
            tot_simulate_succ = np.array(epoch_result['simulate_succ']).sum()
            avg_time = (time.time() - t0) * 1.0 / cur_num
            avg_iter = np.array(epoch_result['iter'], dtype=float).sum() / cur_num
            avg_depth = np.array(epoch_result['route_lens'], dtype=float).sum() / cur_num
            WRITER.add_scalar('prior_loss', prior_loss, tot_num)
            WRITER.add_scalar('critic_loss', critic_loss, tot_num)
            WRITER.add_scalar('actor_loss', actor_loss, tot_num)
            # WRITER.add_scalar('action_entropy', action_entropy, tot_num)
            WRITER.add_scalar('training/success rate', np.array(batch_result['succ']).sum()/len(batch_result['succ']), tot_num)
            WRITER.add_scalar('training/simulate success rate', np.array(batch_result['simulate_succ']).sum()/len(batch_result['simulate_succ']), tot_num)
            WRITER.add_scalar('training/depth', np.array(batch_result['route_lens']).sum()/len(batch_result['route_lens']), tot_num)
            WRITER.add_scalar('training/model calls', np.array(batch_result['iter']).sum()/len(batch_result['iter']), tot_num)
            WRITER.add_scalar('training/route_shortest_cost', np.array(batch_result['route_shortest_cost']).sum()/len(batch_result['route_shortest_cost']), tot_num)
            # print(batch_result['route_shortest_cost'])           
            logging.info('Succ: %d/%d/%d | avg time: %.2f s | avg iter: %.2f | avg depth: %.2f'%
                        (tot_succ, cur_num, num_targets, avg_time, avg_iter, avg_depth))
            logging.info('Simulate Succ: %d|%d|%d' %
                        (tot_simulate_succ, cur_num, num_targets))
            logging.info('Shortest route cost: %.2f | %d' % 
                        (np.array(batch_result['route_shortest_cost']).sum()/len(batch_result['route_shortest_cost']), len(batch_result['route_shortest_cost'])))         
            logging.info('Unsuccessful reactions/Successful molecules: %d/%d' %
                        (epoch_result['num_take_unsucc_when_succ'], epoch_result['num_succ_before_take']))
            logging.info('Building blocks: %d | Unexpanded: %d | Deadends: %d' %
                        (epoch_result["num_building_blocks"], epoch_result["num_unexpanded"], epoch_result["num_deadends"]))
            logging.info(f"Current learning rate is {trainer.lr}")

            with open(args.result_folder + 'timer.json', 'w') as f:
                json.dump(timer, f)

            # save policy and value function
            if args.save_model and tot_num - last_save_num >= args.save_interval:
                saved_model = args.result_folder + '%d_mols/' % (tot_num) + 'one_step/'
                os.makedirs(saved_model, exist_ok=True)
                torch.save(one_step.net.state_dict(), saved_model+'rollout_model.ckpt')
                torch.save(value_net.state_dict(), saved_model+'value_fn.ckpt')
                torch.save(prior_net.state_dict(), saved_model+'prior_fn.ckpt')
                logging.info('Models saved into %s' % saved_model)
                last_save_num = tot_num

        # global_result['succ_epoch_{}'.format(epoch+1)] = epoch_result['succ']
        # global_result['simulate_succ_epoch_{}'.format(epoch+1)] = epoch_result['simulate_succ']
        global_result['simulate_fail_idx_epoch_{}'.format(epoch+1)] = sorted(epoch_result["simulate_fail_idx"])
        global_result['num_simulate_fail_idx_epoch_{}'.format(epoch+1)] = len(epoch_result["simulate_fail_idx"])
        with open(args.result_folder + 'global_result.json', 'w') as f:
            json.dump(global_result, f, sort_keys=True)

    runner.close()
    WRITER.close()
    logging.info('The runner closes gracefully')

# TODO: use torch.nn.DataParallel
def load_model_to_runner(model_net, value_net, prior_net, runner, n_gpus=1):
    if n_gpus > 1:
        # load model parameters to subprocesses using different gpus
        model_net.cpu()
        value_net.cpu()
        prior_net.cpu()
        runner.load_model(model_net.state_dict(), prior_net.state_dict(), value_net.state_dict())
        model_net.cuda()
        value_net.cuda()
        prior_net.cuda()
    else:
        runner.load_model(model_net.state_dict(), prior_net.state_dict(), value_net.state_dict())

if __name__ == '__main__':
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    setup_logger('plan.log')

    retro_mcts()
