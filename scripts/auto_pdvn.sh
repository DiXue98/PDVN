RESULT_FOLDER=./results/pdvn/run2/
TEST_ROUTES=dataset/routes_possible_test_hard.pkl
PLAN_INFO=retro_routes_possible_test_hard

cd retro_star
python retro_mcts.py \
--name 0428_minibs128 \
--starting_molecules dataset/origin_dict_canonical.csv \
--lr 1e-3 --minibatch_size 128 --batch_size 1024 \
--realistic_filter \
--n_epochs 3 --iteration 500 --num_simulations 100 --depth 15 --PUCT_coef 1.0 \
--runner parallel --gpu 3 --n_gpus 4 --n_processes 20 \
--save_model --save_interval 25000 --result_folder ${RESULT_FOLDER} \
--train_routes dataset/routes_train.pkl \
--test_interval 50000000000 --test_batch_size 128

python test_plan.py \
--method retro \
--starting_molecules dataset/origin_dict_canonical.csv \
--realistic_filter \
--iteration 500 --gpu 0 \
--runner parallel --n_gpus 4 --n_processes 20 \
--result_folder ${RESULT_FOLDER} \
--test_routes ${TEST_ROUTES}

python plan_results_analyzer.py \
--result_folder ${RESULT_FOLDER} \
--plan_info ${PLAN_INFO}