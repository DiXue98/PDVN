RESULT_FOLDER=./results/retro_star/chembl/
TEST_ROUTES=dataset/chembl_1000.pkl
# TEST_ROUTES=dataset/gdb17_1000.pkl
BACKWARD_MODEL=<REPLANCE_THIS_WITH_BEST_MODEL>

cd retro_star
python retro_plan.py \
  --test_routes ${TEST_ROUTES} \
  --mlp_model_dump ${BACKWARD_MODEL} \
  --result_folder ${RESULT_FOLDER} \
  --iteration 500