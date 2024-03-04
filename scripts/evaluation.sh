RESULT_FOLDER=./results/pdvn/run1/
TEST_ROUTES=dataset/routes_possible_test_hard.pkl

cd retro_star
python test_plan.py \
--method retro \
--starting_molecules dataset/origin_dict_canonical.csv \
--realistic_filter \
--iteration 500 --gpu 0 \
--runner parallel --n_gpus 4 --n_processes 20 \
--result_folder ${RESULT_FOLDER} \
--test_routes ${TEST_ROUTES}