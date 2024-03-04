RESULT_FOLDER=./results/pdvn/run1/
PLAN_INFO=retro_routes_possible_test_hard

cd retro_star
python plan_results_analyzer.py \
--result_folder ${RESULT_FOLDER} \
--plan_info ${PLAN_INFO}