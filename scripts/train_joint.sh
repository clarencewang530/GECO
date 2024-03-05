#!/bin/bash
#SBATCH -p gpu22
#SBATCH -t 5:00:00
#SBATCH -o /scratch/inf0/user/lliu/chenwang/slurm_logs/slurm-%j.out
#SBATCH --gres gpu:a100:2

DATA_PATH=/scratch/inf0/user/lliu/chenwang/lvis_sample
RESUME_PKL=/scratch/inf0/user/lliu/chenwang/Zero123PlusGS/network-snapshot-020000.pkl
RESUME=/scratch/inf0/user/lliu/chenwang/Zero123PlusGS/zero123plus_vsd/data/model.safetensors

accelerate launch --config_file acc_configs/gpu2.yaml main.py big --workspace logs-joint/ --num_input_views 6 --model_type LGM --num_epochs 10 --bg 0.5 --resume ${RESUME} --batch_size 1 --lr 1e-6 --save_freq 10 --eval_freq 5 --stage 2 --data_path ${DATA_PATH} --model_type Zero123PlusGaussian --num_views 12 --resume_pkl ${RESUME_PKL}