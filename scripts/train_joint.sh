accelerate launch --config_file acc_configs/gpu1.yaml main.py big --workspace logs-joint/ --num_input_views 6 --model_type LGM --num_epochs 1 --bg 0.5 --resume /mnt/kostas-graid/sw/envs/chenwang/workspace/instant123/logs/ft_gray_0.5bg/LGM-20240222-212145/model.safetensors --batch_size 4 --lr 1e-5 --save_freq 10 --eval_freq 5 --stage 2 --data_path /mnt/ds_cache/chenwang/lvis_lgm_render --model_type Zero123PlusGaussian --num_views 12