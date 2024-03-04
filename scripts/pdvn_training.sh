EXP_NAME=pdvn
RESULT_FOLDER=./results/pdvn/run1/

cd retro_star
python retro_mcts.py \
--name ${EXP_NAME} \
--starting_molecules dataset/origin_dict_canonical.csv \
--lr 1e-3 --minibatch_size 128 --batch_size 1024 \
--realistic_filter \
--n_epochs 3 --iteration 500 --num_simulations 100 --depth 15 --PUCT_coef 1.0 \
--runner parallel --gpu 3 --n_gpus 3 --n_processes 15 \
--save_model --save_interval 25000 --result_folder ${RESULT_FOLDER} \
--train_routes dataset/routes_train.pkl \
--test_interval 5000 --test_batch_size 128