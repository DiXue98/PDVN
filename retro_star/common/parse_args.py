import argparse
import os
import sys


parser = argparse.ArgumentParser()

# ===================== gpu id ===================== #
parser.add_argument('--gpu', type=int, default=-1)

# =================== random seed ================== #
parser.add_argument('--seed', type=int, default=1234)

# ==================== dataset ===================== #
parser.add_argument('--train_routes',
                    default='dataset/routes_train.pkl')
parser.add_argument('--test_routes',
                    default='dataset/routes_possible_test_hard.pkl')
parser.add_argument('--starting_molecules', default='dataset/origin_dict_canonical.csv')

# ================== value dataset ================= #
parser.add_argument('--value_root', default='dataset')
parser.add_argument('--value_train', default='train_mol_fp_value_step')
parser.add_argument('--value_val', default='val_mol_fp_value_step')

# ================== one-step model ================ #
parser.add_argument('--mlp_model_dump',
                    default='one_step_model/saved_rollout_state_1_2048.ckpt')
parser.add_argument('--mlp_templates',
                    default='one_step_model/template_rules_1.dat')

parser.add_argument('--mlp_model_dump_A',
                    default='one_step_model/saved_rollout_state_1_2048.ckpt')
parser.add_argument('--mlp_model_dump_B',
                    default='one_step_model/saved_rollout_state_1_2048.ckpt')

parser.add_argument("--sample_mode", type=str,
                    default='template')

# ===================== all algs =================== #
parser.add_argument('--iterations', type=int, default=500)
parser.add_argument('--expansion_topk', type=int, default=50)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--draw_mols', action='store_true')
parser.add_argument('--viz_dir', default='viz')
parser.add_argument('--name', type=str)

# ===================== model ====================== #
parser.add_argument('--fp_dim', type=int, default=2048)
parser.add_argument('--n_layers', type=int, default=1)
parser.add_argument('--latent_dim', type=int, default=128)

# ==================== training ==================== #
parser.add_argument('--n_epochs', type=int, default=3)
parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--minibatch_size', type=int, default=1024)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--save_epoch_int', type=int, default=1)
parser.add_argument('--save_folder', default='saved_models')
parser.add_argument('--train_from', default='')
# parser.add_argument('--advantage', action='store_true')
parser.add_argument('--advantage', type=bool, default=True)
parser.add_argument('--save_model', action='store_true')
parser.add_argument('--save_interval', type=int, default=25000)
parser.add_argument('--entropy_coef', type=float, default=0.01)
parser.add_argument('--gradient_clip', type=float, default=0.5)
parser.add_argument('--realistic_filter', action='store_true')
parser.add_argument('--extra_lesson', action='store_true')
parser.add_argument('--dropout', action='store_true')

# ==================== planning ==================== #
parser.add_argument('--use_single_network', action='store_true')
parser.add_argument('--num_simulations', type=int, default=100)
parser.add_argument('--PUCT_coef', type=float, default=1.0)
parser.add_argument('--depth', type=int, default=15)

# ==================== testing ==================== #
parser.add_argument('--test_interval', type=int, default=5000)
parser.add_argument('--test_batch_size', type=int, default=1024)

# ==================== evaluation =================== #
parser.add_argument('--use_value_fn', action='store_true')
parser.add_argument('--value_model', default='best_epoch_final_4.pt')
parser.add_argument('--result_folder', default='results')
parser.add_argument("--method", type=str, default="mcts")

# ==================== Multiprocessing =================== #
parser.add_argument("--n_proc", default=1, type=int)
parser.add_argument('--runner', type=str, default='parallel')
parser.add_argument('--n_processes', type=int, default=4)
parser.add_argument('--n_gpus', type=int, default=1)

# =================================#
parser.add_argument("--use-r", action='store_true')
parser.add_argument("--save_train_data_folder", type=str)
parser.add_argument("--save_raw_data_folder", type=str)
parser.add_argument("--max_succes_count", type=int, default=1)
parser.add_argument("--fast_mode", action='store_true')
parser.add_argument("--search_bsz", type=int, default=1)
parser.add_argument("--cluster_method", type=str, default="random")
parser.add_argument("--n_clusters", type=int, default=1)

# ======== aug =================== #
parser.add_argument('--forward_model', type=str)
parser.add_argument('--fw_backward_model', type=str)
parser.add_argument('--aug_topk', type=int, default=5)
parser.add_argument('--fw_backward_validate', action='store_true')

# ======= GNN =================== #
parser.add_argument('--use_gnn_plan', action='store_true')
parser.add_argument('--gnn_ckpt', type=str)
parser.add_argument('--gnn_dim', type=int)
parser.add_argument('--gnn_dropout', type=float)
parser.add_argument('--gnn_layers', type=int)
parser.add_argument('--gnn_ratio', type=float)


args = parser.parse_args()

# setup device
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
