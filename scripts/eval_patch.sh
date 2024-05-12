export MODEL_FOLDER='/home/guest/Desktop/RVT/rvt/runs/patch/12x12patch_4_100_feat256_upsample4'
export BEST1_MODEL_PTH='model_39_best.pth'
export BEST2_MODEL_PTH='model_48_best.pth'
export TASK=all # ['close_grill','open_fridge','open_grill','open_oven','put_umbrella_in_umbrella_stand','get_ice_from_fridge','turn_oven_on']
export TASK_NUM=30
export EVAL_FOLDER='/home/guest/Desktop/RVT/data/test'
#  python eval_patch.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take1 --headless --device 0 --model-name model_last.pth --episode-length 80
#  python eval_patch.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take2 --headless --device 0 --model-name model_last.pth --episode-length 80
#  python eval_patch.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take3 --headless --device 0 --model-name model_last.pth --episode-length 80

#  python eval_patch.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/best1_take1 --headless --device 0 --model-name $BEST1_MODEL_PTH --episode-length 80
#  python eval_patch.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/best1_take2 --headless --device 0 --model-name $BEST1_MODEL_PTH --episode-length 80
#  python eval_patch.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/best1_take3 --headless --device 0 --model-name $BEST1_MODEL_PTH --episode-length 80
 
#  python eval_patch.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/best2_take2 --headless --device 0 --model-name $BEST2_MODEL_PTH --episode-length 80
#  python eval_patch.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/best2_take3 --headless --device 0 --model-name $BEST2_MODEL_PTH --episode-length 80
#  python eval_patch.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/best2_take1 --headless --device 0 --model-name $BEST2_MODEL_PTH --episode-length 80


export TASK_NUM=20
export EVAL_FOLDER='/home/guest/Desktop/RVT/data/test_hard'
 python eval_patch.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take1_hard2 --headless --device 0 --model-name model_last.pth --episode-length 80
 python eval_patch.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take2_hard2 --headless --device 0 --model-name model_last.pth --episode-length 80
 python eval_patch.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take3_hard2 --headless --device 0 --model-name model_last.pth --episode-length 80
