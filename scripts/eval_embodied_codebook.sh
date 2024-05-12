
export MODEL_FOLDER='/home/guest/Desktop/RVT/rvt/runs/codebook/6x6codebook_4_100_feat256_on_rgb'
export TASK_NUM=20
export TASK=all
export BEST1_MODEL_PTH='model_21.pth'
export BEST2_MODEL_PTH='model_best2.pth'
# python eval_embodied_codebook.py --model-folder $MODEL_FOLDER --eval-datafolder /home/guest/Desktop/RVT/data/test_hard  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take1_hard2 --headless --device 0 --model-name model_last.pth --episode-length 80
python eval_embodied_codebook.py --model-folder $MODEL_FOLDER --eval-datafolder /home/guest/Desktop/RVT/data/test_hard  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take2_hard2 --headless --device 0 --model-name model_last.pth --episode-length 80
python eval_embodied_codebook.py --model-folder $MODEL_FOLDER --eval-datafolder /home/guest/Desktop/RVT/data/test_hard  --tasks $TASK --eval-episodes $TASK_NUM --log-name test/last_take3_hard2 --headless --device 0 --model-name model_last.pth  --episode-length 80
