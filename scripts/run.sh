#!/bin/bash
#SBATCH -p gpu22
#SBATCH -t 12:00:00
#SBATCH -o /scratch/inf0/user/lliu/chenwang/slurm_logs/slurm-%j.out
#SBATCH --gres gpu:a40:1

CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.run --standalone --nproc_per_node=gpu sample_data.py big --resume /scratch/inf0/user/lliu/chenwang/Zero123PlusGS/zero123plus_vsd/data/model.safetensors --pipeline zero123plus --sample_bs 6 --sample_nv 50 --sample_data_path /scratch/inf0/user/lliu/lvis_six_random --sample_output_path /scratch/inf0/user/lliu/chenwang/lvis_sample --sample_start 36000 --sample_end 39000