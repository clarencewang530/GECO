python train.py --outdir=./training-runs/zero123 --gen=LRM --arch=unet_zero123  --data='zero123:32:0' \
 --glr=0.0005 --loss_type=vsd --cfg_tchr=3.0 --cfg_stu=1.0 \
  --max_iters=5000 --suffix 'debug' --batch=4 --gpus=1 \
  --lr=1e-5 --snap=10000 --img-snap=50 --timestamp=True --ft_all=False --model_path='bennyguo/zero123-xl-diffusers' --custom_pipeline='./training/zero123.py' --root_dir='/mnt/kostas-graid/sw/envs/chenwang/workspace/lrm-zero123/assets' --batch_pose=1