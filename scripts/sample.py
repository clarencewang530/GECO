import os
import subprocess

all_files = 10
num_gpus = 2




commands = []
for i in range(num_gpus):
    start = i * all_files // num_gpus
    end = (i + 1) * all_files // num_gpus
    command = f'CUDA_VISIBLE_DEVICES={i} python3 -m torch.distributed.run --standalone --nproc_per_node=gpu sample_data.py big --resume /mnt/kostas-graid/sw/envs/chenwang/workspace/instant123/logs/ft_gray_0.5bg/LGM-20240222-022708/model.safetensors --pipeline zero123plus --sample_bs 1 --sample_nv 50 --sample_data_path /mnt/kostas-graid/sw/envs/chenwang/data/lvis_first --sample_output_path /mnt/kostas-graid/sw/envs/chenwang/workspace/instant123/sample_output --sample_start {start} --sample_end {end} &'
    os.system(command)
    print(command)
    # commands.append(command)