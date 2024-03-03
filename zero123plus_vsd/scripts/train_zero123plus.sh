#!/bin/bash
#SBATCH -p gpu22
#SBATCH -t 1:00:00
#SBATCH -o /scratch/inf0/user/lliu/chenwang/slurm_logs/slurm-%j.out
#SBATCH --gres gpu:a100:1
python3 train.py --outdir=./training-runs/zero123plus-dtype --data='zero123plus:80:1' --cond=0 --gen=same --arch=unet_zero123plus \
 --glr=1e-5 --loss_type=vsd --cfg_tchr=4.0 --cfg_stu=1.0 \
  --max_iters=20000 --suffix '800_vsdtrianlgm' --batch=4 --gpus=1 \
  --lr=1e-6 --snap=500 --img-snap=50 --timestamp=True --ft_all=True --model_path='sudo-ai/zero123plus-v1.1' --custom_pipeline='../pipelines/zero123plus.py' --init_t=950 --dtype=fp32 --root_dir=/scratch/inf0/user/lliu/chenwang/lvis_sample --use_lgm=True --use_vsd=False \
  --use_reg=True --resume=/scratch/inf0/user/lliu/chenwang/Zero123PlusGS/network-snapshot-020000.pkl --reg_data_path=/scratch/inf0/user/lliu/chenwang/lvis_sample
