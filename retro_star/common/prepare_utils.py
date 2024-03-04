import pickle
import pandas as pd
import logging
from mlp_retrosyn.mlp_inference import MLPModel
from retro_star.alg import molstar, molstar_ens, online_mcts

def prepare_starting_molecules(filename):
    logging.info('Loading starting molecules from %s' % filename)

    if filename[-3:] == 'csv':
        starting_mols = set(list(pd.read_csv(filename)['mol']))
    else:
        assert filename[-3:] == 'pkl'
        with open(filename, 'rb') as f:
            starting_mols = pickle.load(f)

    logging.info('%d starting molecules loaded' % len(starting_mols))
    return starting_mols

def prepare_mlp(templates, model_dump, gpu=-1, realistic_filter=False):
    device = 0 if gpu > -1 else -1
    logging.info('Templates: %s' % templates)
    logging.info('Loading trained mlp model from %s' % model_dump)
    one_step = MLPModel(model_dump, templates, device=device, realistic_filter=realistic_filter)
    return one_step

def prepare_molstar_planner(one_step, value_fn, starting_mols, expansion_topk,
                            iterations, viz=False, viz_dir=None, draw_mols=True):
    expansion_handle = lambda x: one_step.run(x, topk=expansion_topk)

    plan_handle = lambda x, y=0: molstar(
        target_mol=x,
        target_mol_id=y,
        starting_mols=starting_mols,
        expand_fn=expansion_handle,
        value_fn=value_fn,
        iterations=iterations,
        viz=viz,
        viz_dir=viz_dir,
        draw_mols=draw_mols,
    )
    return plan_handle


def prepare_molstar_planner_ens(one_step_A, one_step_B, value_fn, starting_mols, expansion_topk,
                                iterations, viz=False, viz_dir=None):
    expansion_handle_A = lambda x: one_step_A.run(x, topk=expansion_topk)
    expansion_handle_B = lambda x: one_step_B.run(x, topk=expansion_topk)

    plan_handle = lambda x, y=0: molstar_ens(
        target_mol=x,
        target_mol_id=y,
        starting_mols=starting_mols,
        expand_fn_A=expansion_handle_A,
        expand_fn_B=expansion_handle_B,
        value_fn=value_fn,
        iterations=iterations,
        viz=viz,
        viz_dir=viz_dir
    )
    return plan_handle


def prepare_onlinemcts_planner(one_step, value_fn, prior_fn, starting_mols, expansion_topk,
                         iterations, args, viz=False, viz_dir=None):
    expansion_handle = lambda x: one_step.run(x, topk=expansion_topk)

    plan_handle = lambda x, y=0, test=False: online_mcts(
        target_mol=x,
        target_mol_id=y,
        starting_mols=starting_mols,
        expand_fn=expansion_handle,
        value_fn=value_fn,
        prior_fn=prior_fn,
        iterations=iterations,
        args=args,
        viz=viz,
        viz_dir=viz_dir,
        test=test,
    )

    return plan_handle