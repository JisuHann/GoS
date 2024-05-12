export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8

torchrun --nproc-per-node=2 --max-restarts=1 train_ddp.py --exp_cfg_path configs/all_patch.yaml --device 0