#!/bin/bash
#SBATCH -p gpu22
#SBATCH -t 4:00:00
#SBATCH -o /scratch/inf0/user/lliu/chenwang/slurm_logs/slurm-%j.out
#SBATCH --gres gpu:a100:1
export LD_LIBRARY_PATH=/scratch/inf0/user/lliu/chenwang/cuda-11.8/lib64:$LD_LIBRARY_PATH
export PATH=/scratch/inf0/user/lliu/chenwang/cuda-11.8/bin:$PATH
accelerate launch --config_file acc_configs/gpu1.yaml main.py big --workspace logs/zero123plus-new-ftall-new --model_type Zero123PlusGaussian