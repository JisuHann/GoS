
export MODEL_FOLDER='/home/guest/Desktop/RVT/rvt/runs/gmask/6x6gmask_4_100_feat256'
export BEST_MODEL_PTH='model_35_best.pth'
export TASK=turn_oven_on
export TASK_NUM=30
export EVAL_FOLDER='/home/guest/Desktop/RVT/data/test'
#  python eval_gmask.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take1 --headless --device 0 --model-name model_last.pth --episode-length 80
#  python eval_gmask.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take2 --headless --device 0 --model-name model_last.pth --episode-length 80
#  python eval_gmask.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take3 --headless --device 0 --model-name model_last.pth  --episode-length 80

#  python eval_gmask.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/best1_take1 --headless --device 0 --model-name $BEST_MODEL_PTH --episode-length 80
#  python eval_gmask.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/best1_take2 --headless --device 0 --model-name $BEST_MODEL_PTH --episode-length 80
#  python eval_gmask.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/best1_take3 --headless --device 0 --model-name $BEST_MODEL_PTH --episode-length 80
 

export TASK_NUM=20
export EVAL_FOLDER='/home/guest/Desktop/RVT/data/test_hard'
 python eval_gmask.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take1_dist_hard2 --headless --device 0 --model-name model_last.pth --episode-length 80
 python eval_gmask.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take2_dist_hard2 --headless --device 0 --model-name model_last.pth --episode-length 80
 python eval_gmask.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take3_dist_hard2 --headless --device 0 --model-name model_last.pth  --episode-length 80

export TASK=open_grill
 python eval_gmask.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take2_dist_hard2 --headless --device 0 --model-name model_last.pth --episode-length 80
 python eval_gmask.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take3_dist_hard2 --headless --device 0 --model-name model_last.pth  --episode-length 80

export TASK=close_grill
 python eval_gmask.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take2_dist_hard2 --headless --device 0 --model-name model_last.pth --episode-length 80
 python eval_gmask.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take3_dist_hard2 --headless --device 0 --model-name model_last.pth  --episode-length 80

#  python eval_gmask.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/best1_take1_dist --headless --device 0 --model-name $BEST_MODEL_PTH --episode-length 80
#  python eval_gmask.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/best1_take2_dist --headless --device 0 --model-name $BEST_MODEL_PTH --episode-length 80
#  python eval_gmask.py --model-folder $MODEL_FOLDER --eval-datafolder $EVAL_FOLDER  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/best1_take3_dist --headless --device 0 --model-name $BEST_MODEL_PTH --episode-length 80
 