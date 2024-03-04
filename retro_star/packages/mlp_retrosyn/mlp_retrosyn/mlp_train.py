import os
from collections import defaultdict
from tqdm import tqdm
from .mlp_policies import train_mlp
from pprint import pprint

import numpy as np
import torch
import random

if __name__ == '__main__':
    import  argparse
    parser = argparse.ArgumentParser(description="train function for retrosynthesis Planner policies")
    parser.add_argument('--template_path',
                        default= "",
                        type=str, help='Specify the path of the template.data')
    parser.add_argument('--template_path_test',
                        default= "",
                        type=str, help='Specify the path of the template.data')

    # parser.add_argument('--template_path',default= 'data/cooked_data/templates.dat',
    #                     type=str, help='Specify the path of the template.data')
    # parser.add_argument('--template_path_test', default='data/cooked_data/templates.dat',
    #                     type=str, help='Specify the path of the template.data')
    parser.add_argument('--template_rule_path', default='data/cooked_data/template_rules_1.dat',
                        type=str, help='Specify the path of all template rules.')
    parser.add_argument('--model_dump_folder',default='./model',
                        type=str, help='specify where to save the trained models')
    parser.add_argument('--fp_dim',default=2048, type=int,
                        help="specify the fingerprint feature dimension")
    parser.add_argument('--batch_size', default=1024, type=int,
                        help="specify the batch size")
    parser.add_argument('--dropout_rate', default=0.4, type=float,
                        help="specify the dropout rate")
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help="specify the learning rate")
    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--train_path', default="",
                        type=str)
    parser.add_argument('--test_path', default="",
                        type=str)

    # parser.add_argument('--train_path', default='../data/uspto_all/proc_train_cano_smiles_w_tmpl.csv',
    #                     type=str)
    # parser.add_argument('--test_path', default='../data/uspto_all/proc_test_cano_smiles_w_tmpl.csv',
    #                     type=str)
    parser.add_argument("--train_from", default='',
                        type=str)

    # Learning without Forgetting/Knowledge distillation
    parser.add_argument("--lwf", action="store_true")
    parser.add_argument("--old_model", default='',
                        type=str)
    parser.add_argument("--alpha", "-a", default=1.,
                        type=float)
    parser.add_argument("--temp", "-t", default=1.,
                        type=float)

    # Train forward model
    parser.add_argument("--forward", action="store_true")

    # Seed
    parser.add_argument("--seed", default=0, type=int)

    args = parser.parse_args()

    if args.seed != 0:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        random.seed(args.seed)

    template_path = args.template_path
    template_rule_path = args.template_rule_path
    model_dump_folder = args.model_dump_folder
    fp_dim = args.fp_dim
    batch_size = args.batch_size
    dropout_rate = args.dropout_rate
    lr = args.learning_rate
    print('Loading data...')
    prod_to_rules = defaultdict(set)
    ### read the template data.
    with open(template_path, 'r') as f:
        for l in tqdm(f, desc="reading the mapping from prod to rules"):
            rule, prod = l.strip().split('\t')
            prod_to_rules[prod].add(rule)

    prod_to_rules_test = defaultdict(set)
    with open(args.template_path_test, 'r') as f:
        for l in tqdm(f, desc="reading the mapping from prod to rules"):
            rule, prod = l.strip().split('\t')
            prod_to_rules_test[prod].add(rule)

    if not os.path.exists(model_dump_folder):
        os.mkdir(model_dump_folder)
    if args.lwf:
        if args.old_model == "":
            args.old_model = args.train_from
    pprint(args)
    train_mlp(prod_to_rules,
              prod_to_rules_test,
              template_rule_path,
              fp_dim=fp_dim,
              batch_size=batch_size,
              lr=lr,
              epochs=args.epochs,
              dropout_rate=dropout_rate,
              saved_model=os.path.join(model_dump_folder, 'saved_rollout_state_1'),
              train_path=args.train_path,
              test_path=args.test_path,
              train_from=args.train_from,
              lwf=args.lwf,
              old_model=args.old_model,
              alpha=args.alpha,
              temp=args.temp,
              backward=not args.forward)
