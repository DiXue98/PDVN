import os
import numpy as np
from retro_star.common import prepare_starting_molecules, prepare_mlp, \
                              prepare_molstar_planner, prepare_onlinemcts_planner
from retro_star.model import ValueMLP, PriorMLP
from mlp_retrosyn.mlp_policies import preprocess

import torch
import torch.multiprocessing as mp
from multiprocessing import Manager
from multiprocessing import Queue
from time import sleep
from viz_tree.mol_tree_to_paroutes_dict import mol_tree_to_paroutes_dict

mp.set_start_method('spawn', force=True)

class SerialRunner:
    def __init__(self, args):
        self.args = args
        device = torch.device('cuda' if args.n_gpus > 0 else 'cpu')

        # load one-step model
        one_step = prepare_mlp(args.mlp_templates, args.mlp_model_dump,
                               gpu=args.gpu, realistic_filter=args.realistic_filter)
        starting_mols = prepare_starting_molecules(args.starting_molecules)
        self.model_net = one_step.net

        # prepare value function
        self.value_net = ValueMLP(
            n_layers=args.n_layers,
            fp_dim=args.fp_dim,
            latent_dim=args.latent_dim,
            dropout_rate=0.1,
            device=device,
        ).to(device)
        self.value_net.eval()

        if args.use_value_fn:
            def value_fn(mol):
                fp = preprocess(mol, fp_dim=args.fp_dim).reshape(1,-1)
                fp = torch.FloatTensor(fp).to(device)
                v = self.value_net(fp).item()
                return v
        else:
            value_fn = lambda x: 0

        # prepare prior function
        self.prior_net = PriorMLP(
            n_layers=args.n_layers,
            fp_dim=args.fp_dim,
            latent_dim=args.latent_dim,
            dropout_rate=0.1,
            device=device,
        ).to(device)
        self.prior_net.eval()

        def prior_fn(mol):
            fp = preprocess(mol, fp_dim=args.fp_dim).reshape(1,-1)
            fp = torch.FloatTensor(fp).to(device)
            p = self.prior_net(fp).item()
            return p

        # setup planners
        self.onlinemcts_handle = prepare_onlinemcts_planner(
                one_step=one_step,
                value_fn=value_fn,
                # value_fn=lambda x:0,
                prior_fn=prior_fn,
                starting_mols=starting_mols,
                expansion_topk=args.expansion_topk,
                iterations=args.iterations,
                args=args,
                viz=args.viz,
                viz_dir=args.viz_dir
            )

        self.molstar_handle = prepare_molstar_planner(
            one_step=one_step,
            value_fn=value_fn,
            starting_mols=starting_mols,
            expansion_topk=args.expansion_topk,
            iterations=args.iterations,
            viz=args.viz,
            viz_dir=args.viz_dir,
            draw_mols=args.draw_mols,
        )

    def run(self, target_mols, target_mol_ids, test=False):
        batch_result = {
            'ids': [],
            'succ': [],
            'iter': [],
            'route_costs': [],
            'route_lens': [],
        }
        mols_batch, vals_batch, tmps_batch = [], [], []
        for mol_id, target_mol in zip(target_mol_ids, target_mols):
            # succ, (mol_tree, iter) = self.rollout_handle(target_mol, mol_id, test=test)
            succ, (mol_tree, iter) = self.rollout_handle(target_mol, mol_id)
            # generate training data
            mols, vals, tmps = mol_tree.root.generate_training_dataset()
            mols_batch += mols
            vals_batch += vals
            tmps_batch += tmps

            # record batch results
            batch_result['ids'].append(mol_id)
            batch_result['succ'].append(succ)
            batch_result['iter'].append(iter)
            batch_result['route_costs'].append(mol_tree.value)
            batch_result['route_lens'].append(mol_tree.depth)

        return mols_batch, vals_batch, tmps_batch, batch_result

    def mcts(self, target_mols, target_mol_ids, test=False):
        batch_result = {
            "ids": [],
            "succ": [],
            "simulate_succ": [],
            "iter": [],
            "route_lens": [],
            "num_succ_before_take": 0,
            "num_take_unsucc_when_succ": 0,
            "num_deadends": 0,
            "num_building_blocks": 0,
            "num_unexpanded": 0,
            "route_shortest_cost": [],
            "mol_nodes": [],
            "reaction_nodes": [],
            "nll_reference": [], # negative log likelihood given by the reference model
            "nll": [], # negative log likelihood given by the trained model
            "succ_len": [],
        }
        mols_r_batch, pris_batch, mols_v_batch, vals_batch, mols_t_batch, tmps_batch = [], [], [], [], [], []
        for mol_id, target_mol in zip(target_mol_ids, target_mols):
            succ, (mol_tree, iter), result = self.onlinemcts_handle(target_mol, mol_id, test=test)
            mols_r_batch += result["mols_r"]
            pris_batch += result["pris"]
            mols_v_batch += result["mols_v"]
            vals_batch += result["vals"]
            mols_t_batch += result["mols_t"]
            tmps_batch += result["tmps"]

            # record batch results
            batch_result['ids'].append(mol_id)
            batch_result['succ'].append(succ)
            batch_result['simulate_succ'].append(mol_tree.simulate_succ)
            batch_result['iter'].append(iter)
            batch_result['route_lens'].append(mol_tree.depth)
            batch_result["num_succ_before_take"] += result["num_succ_before_take"]
            batch_result["num_take_unsucc_when_succ"] += result["num_take_unsucc_when_succ"]
            batch_result["num_deadends"] += result["num_deadends"]
            batch_result["num_building_blocks"] += result["num_building_blocks"]
            batch_result["num_unexpanded"] += result["num_unexpanded"]
            batch_result['mol_nodes'].append(len(mol_tree.mol_nodes))
            batch_result['reaction_nodes'].append(len(mol_tree.reaction_nodes))
            batch_result['nll'].append(mol_tree.root.cumulative_nll)
            batch_result['nll_reference'].append(mol_tree.root.cumulative_nll_reference)
            batch_result['succ_len'].append(mol_tree.root.succ_len)
            if mol_tree.root.succ_len != np.inf:
                batch_result["route_shortest_cost"].append(mol_tree.root.succ_len)

        return mols_r_batch, pris_batch, mols_v_batch, vals_batch, mols_t_batch, tmps_batch, batch_result

    def plan(self, target_mols, target_mol_ids):
        mols_batch, vals_batch, tmps_batch = [], [], []
        batch_result = {
            'ids': [],
            'succ': [],
            'iter': [],
            'route_costs': [],
            'route_lens': [],
        }
        for mol_id, target_mol in zip(target_mol_ids, target_mols):
            # generate training data
            succ, (mols, vals, tmps) = self.retrostar_handle(target_mol, mol_id)
            mols_batch += mols
            vals_batch += vals
            tmps_batch += tmps

            # record batch results
            batch_result['ids'].append(mol_id)
            batch_result['succ'].append(succ)

        return mols_batch, vals_batch, tmps_batch, batch_result

    def retro(self, target_mols, target_mol_ids):
        batch_result = {
            "ids": [],
            "succ": [],
            "iter": [],
            "mol_nodes": [],
            "reaction_nodes": [],
            "nll_reference": [], # negative log likelihood given by the reference model
            "nll": [], # negative log likelihood given by the trained model
            "succ_len": [],
        }
        for mol_id, target_mol in zip(target_mol_ids, target_mols):
            # generate training data
            succ, msg = self.molstar_handle(target_mol, mol_id)

            # record batch results
            batch_result['ids'].append(mol_id)
            batch_result['succ'].append(succ)
            batch_result['iter'].append(msg[1])
            batch_result['mol_nodes'].append(len(msg[2].mol_nodes))
            batch_result['reaction_nodes'].append(len(msg[2].reaction_nodes))     
            if succ:
                # save successful route
                succ_route = mol_tree_to_paroutes_dict(msg[2].root)
                # f = open(self.args.result_folder + 'routes_dict/' + 'mol_{}.json'.format(mol_id), 'w')
                # json.dump(succ_route, f, indent=4)
                # f.close()

                batch_result['nll_reference'].append(succ_route["value_reference"])
                batch_result['nll'].append(succ_route["succ_value"])
                batch_result['succ_len'].append(msg[0].length)
            else:
                batch_result['nll_reference'].append(0)
                batch_result['nll'].append(0)
                batch_result['succ_len'].append(0)

        return batch_result

    def close(self):
        pass

    def load_model(self, model_state_dict, prior_state_dict=None, value_state_dict=None):
        self.model_net.load_state_dict(model_state_dict)
        if value_state_dict:
            self.value_net.load_state_dict(value_state_dict)
        if prior_state_dict:
            self.prior_net.load_state_dict(prior_state_dict)

def worker(q1, remote, args, gpu=-1):
    if gpu > -1:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    local_runner = SerialRunner(args)
    while True:
        cmd = remote.recv()
        if type(cmd) != str: 
            assert len(cmd) == 2  # for load model mode.
            cmd, data = cmd[0], cmd[1]

        if cmd == "load":
            local_runner.load_model(*data)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd in ["sample", "test", "plan", "retro", "mcts", "mcts_test"]:
            results = []
            # while not q1.empty(): # sometimes get stuck here.
            while True:
                element = q1.get()
                if element == 'DONE': # stop element
                    break
                target_mols, target_mol_ids = [element[0]], [element[1]]
                if cmd == 'sample':
                    mols_batch, vals_batch, tmps_batch, batch_result = local_runner.run(target_mols, target_mol_ids)
                    results.append([mols_batch, vals_batch, tmps_batch, batch_result])
                elif cmd == 'test':
                    mols_batch, vals_batch, tmps_batch, batch_result = local_runner.run(target_mols, target_mol_ids, test=True)
                    results.append([mols_batch, vals_batch, tmps_batch, batch_result])
                elif cmd == 'retro':
                    batch_result = local_runner.retro(target_mols, target_mol_ids)
                    results.append([batch_result])
                elif cmd == 'plan':
                    mols_batch, vals_batch, tmps_batch, batch_result = local_runner.plan(target_mols, target_mol_ids)
                    results.append([mols_batch, vals_batch, tmps_batch, batch_result])
                elif cmd == 'mcts' or cmd == 'mcts_test':
                    mols_r_batch, pris_batch, mols_v_batch, vals_batch, mols_t_batch, tmps_batch, batch_result = local_runner.mcts(target_mols, target_mol_ids, test=(cmd=='mcts_test'))
                    results.append([mols_r_batch, pris_batch, mols_v_batch, vals_batch, mols_t_batch, tmps_batch, batch_result])
            remote.send(results) # send data to continue main process. 
        else:
            raise NotImplementedError


class ParallelRunner:
    def __init__(self, args):
        self.args = args
        self.n_gpus = args.n_gpus
        self.n_processes = args.n_processes
        self.parent_conns, self.worker_conns = zip(*[mp.Pipe() for _ in range(self.n_processes)])

        self.manager = Manager()
        self.q1 = self.manager.Queue() # use q1 to pass the batch target mols from main process to workers. 
        # directly use Pipe to return batch results from workers to main process.
        self.ps = [mp.Process(target=worker, args=(self.q1, self.worker_conns[i], args, i%self.n_gpus if self.n_gpus else -1,)) for i in range(self.n_processes)]
        
        for p in self.ps:
            p.daemon = True
            p.start()

    def run(self, target_mols, target_mol_ids, test=False):
        # 1. put all the target mols and their id into the shared queue.
        for target_mol, target_mol_id in zip(target_mols, target_mol_ids):
            self.q1.put((target_mol, target_mol_id))

        for i in range(0, self.n_processes):
            self.q1.put('DONE')

        # assert self.q1.qsize() == (self.len(target_mols) + self.n_processes)  # from exp, sometimes will violate. delay in mp.Queue.put() in main process.

        # 2. send message to each worker process to let them start working.
        for i in range(self.n_processes):
            if test:
                self.parent_conns[i].send("test")
            else:
                self.parent_conns[i].send("sample")

        # 3. prepare container to save the results.
        batch_result = {
            'ids': [],
            'succ': [],
            'iter': [],
            'route_costs': [],
            'route_lens': []
        }
        mols_batch, vals_batch, tmps_batch = [], [], []

        # 4. wait until each worker process passes data.
        for i in range(self.n_processes):
            results = self.parent_conns[i].recv()
            for result in results:
                mols_batch += result[0]
                vals_batch += result[1]
                tmps_batch += result[2]        
                for key in batch_result.keys():
                    batch_result[key] += (result[3][key])
                    
        return mols_batch, vals_batch, tmps_batch, batch_result

    def mcts(self, target_mols, target_mol_ids, test=False):
        # 1. put all the target mols and their id into the shared queue.
        for target_mol, target_mol_id in zip(target_mols, target_mol_ids):
            self.q1.put((target_mol, target_mol_id))

        for i in range(0, self.n_processes):
            self.q1.put('DONE')

        assert self.q1.qsize() == (len(target_mols) + self.n_processes)

        # 2. send message to each worker process to let them start working.
        for i in range(self.n_processes):
            if test:
                self.parent_conns[i].send("mcts_test")
            else:
                self.parent_conns[i].send("mcts")

        # 3. prepare container to save the results.
        batch_result = {
            "ids": [],
            "succ": [],
            "simulate_succ": [],
            "iter": [],
            "route_lens": [],
            "num_succ_before_take": 0,
            "num_take_unsucc_when_succ": 0,
            "num_deadends": 0,
            "num_building_blocks": 0,
            "num_unexpanded": 0,
            "route_shortest_cost": [],
            "mol_nodes": [],
            "reaction_nodes": [],
            "nll_reference": [],
            "nll": [],
            "succ_len": [],
        }
        mols_r_batch, pris_batch, mols_v_batch, vals_batch, mols_t_batch, tmps_batch = [], [], [], [], [], []

        # 4. wait until each worker process passes data.
        for i in range(self.n_processes):
            results = self.parent_conns[i].recv()
            for result in results:
                mols_r_batch += result[0]
                pris_batch += result[1]
                mols_v_batch += result[2]
                vals_batch += result[3]
                mols_t_batch += result[4]
                tmps_batch += result[5]
                for key in batch_result.keys():
                    batch_result[key] += (result[-1][key])

        return mols_r_batch, pris_batch, mols_v_batch, vals_batch, mols_t_batch, tmps_batch, batch_result

    def retro(self, target_mols, target_mol_ids):
        # 1. put all the target mols and their id into the shared queue.
        for target_mol, target_mol_id in zip(target_mols, target_mol_ids):
            self.q1.put((target_mol, target_mol_id))

        for i in range(0, self.n_processes):
            self.q1.put('DONE')

        assert self.q1.qsize() == (len(target_mols) + self.n_processes)

        # 2. send message to each worker process to let them start working.
        for i in range(self.n_processes):
            self.parent_conns[i].send("retro")

        # 3. prepare container to save the results.
        batch_result = {
            "ids": [],
            "succ": [],
            "iter": [],
            "mol_nodes": [],
            "reaction_nodes": [],
            "nll_reference": [], # negative log likelihood given by the reference model
            "nll": [], # negative log likelihood given by the trained model
            "succ_len": [],
        }

        # 4. wait until each worker process passes data.
        for i in range(self.n_processes):
            results = self.parent_conns[i].recv()
            for result in results:
                for key in batch_result.keys():
                    batch_result[key] += (result[-1][key])

        return batch_result

    def plan(self, target_mols, target_mol_ids):
        # 1. put all the target mols and their id into the shared queue.
        for target_mol, target_mol_id in zip(target_mols, target_mol_ids):
            self.q1.put((target_mol, target_mol_id))

        for i in range(0, self.n_processes):
            self.q1.put('DONE')

        assert self.q1.qsize() == (self.len(target_mols) + self.n_processes)

        # 2. send message to each worker process to let them start working.
        for i in range(self.n_processes):
            self.parent_conns[i].send("plan")

        batch_result = {
            'ids': [],
            'succ': [],
            'simulate_succ': [],
        }
        mols_batch, vals_batch, tmps_batch = [], [], []

        # 4. wait until each worker process passes data.
        for i in range(self.n_processes):
            results = self.parent_conns[i].recv()
            for result in results:
                mols_batch += result[0]
                vals_batch += result[1]
                tmps_batch += result[2]        
                for key in batch_result.keys():
                    batch_result[key] += (result[3][key])

        return mols_batch, vals_batch, tmps_batch, batch_result

    def load_model(self, model_state_dict, prior_state_dict, value_state_dict=None): # TODO
        for i in range(self.n_processes):
            if value_state_dict:
                self.parent_conns[i].send(("load", (model_state_dict, prior_state_dict, value_state_dict)))
            else:
                self.parent_conns[i].send(("load", (model_state_dict, )))

    def close(self):
        for i in range(self.n_processes):
            # self.parent_conns[i].send(('close', None))
            self.parent_conns[i].send('close')