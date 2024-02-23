python train.py --outdir=./training-runs/sd --data='sd:64:0' --cond=0 --gen=unet_sd --arch=unet_sd \
 --glr=1e-6 --loss_type=vsd --cfg_tchr=8.0 \
  --max_iters=20000 --suffix 't872' --batch=1 --gpus=1 \
  --lr=1e-6 --snap=10000 --img-snap=50 --timestamp=True --ft_all=True --model_path='runwayml/stable-diffusion-v1-5' --init_t=872 --prompt_path="/mnt/kostas-graid/sw/envs/chenwang/workspace/diffgan/local_utils/metadata.parquet"
