python3 train.py --outdir=./training-runs/zero123plus-dtype --data='zero123plus:80:1' --cond=0 --gen=same --arch=unet_zero123plus \
 --glr=1e-6 --loss_type=vsd --cfg_tchr=4.0 --cfg_stu=1.0 \
  --max_iters=500000 --suffix '800_fp32' --batch=1 --gpus=1 \
  --lr=1e-6 --snap=5000 --img-snap=5 --timestamp=True --ft_all=True --model_path='sudo-ai/zero123plus-v1.1' --custom_pipeline='../pipelines/zero123plus.py' --init_t=800 --dtype=fp32 --root_dir=/mnt/ds_cache/views_release