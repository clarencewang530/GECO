import os
import tyro
import glob
import imageio
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from safetensors.torch import load_file
import rembg

import kiui
from kiui.op import recenter
from kiui.cam import orbit_camera

import sys
sys.path.append('..')
from core.options import AllConfigs, Options
from core.models_gs import Zero123PlusGaussian
from core.models_lgm import LGM
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, DDPMScheduler
from PIL import Image
import einops
import pickle
import time
import json
import yaml

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

class LGMLoader:
    def __init__(self, device):
        conf = yaml.safe_load(open('conf.yml'))
        opt = tyro.extras.from_yaml(Options, conf)
        opt.output_size = 320
        opt.bg = 0.5
        self.opt = opt
        self.model = LGM(opt)
        # resume pretrained checkpoint
        if opt.resume is not None:
            if opt.resume.endswith('safetensors'):
                ckpt = load_file(opt.resume, device='cpu')
            else:
                ckpt = torch.load(opt.resume, map_location='cpu')
            self.model.load_state_dict(ckpt, strict=False)
            print(f'[INFO] Loaded checkpoint from {opt.resume}')
        else:
            print(f'[WARN] model randomly initialized, are you sure?')

        # device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.float().to(self.device)
        self.model.eval().requires_grad_(False)

        tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        proj_matrix[0, 0] = 1 / tan_half_fov
        proj_matrix[1, 1] = 1 / tan_half_fov
        proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[2, 3] = 1
        self.proj_matrix = proj_matrix
    
    def get_cam_poses(self, elevations, azimuths, batch_gpu):
        cam_poses = [orbit_camera(ele, azi, radius=self.opt.cam_radius, opengl=True) for (ele, azi) in zip(elevations, azimuths)]
        cam_poses = torch.from_numpy(np.stack(cam_poses, axis=0)).to(self.device)
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        cam_view = torch.inverse(cam_poses).transpose(1, 2).unsqueeze(0).repeat(batch_gpu,1,1,1) # [B, V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [B, V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3].unsqueeze(0).repeat(batch_gpu,1,1) # [B, V, 3]
        return cam_view, cam_view_proj, cam_pos
    
    def process_from_zero123plus(self, zero123out, rays_embeddings, cam_view, cam_view_proj, cam_pos, bg_color):
        B = zero123out.shape[0]
        input_image = einops.rearrange((zero123out + 1) / 2, 'b c (h2 h) (w2 w) -> b (h2 w2) c h w', h2=3, w2=2).reshape(-1, 3, 320, 320) # (B*V, 3, H, W)
        input_image = F.interpolate(input_image, size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False)
        input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD).reshape(B, 6, 3, self.opt.input_size, self.opt.input_size) # [1, 4, 3, 256, 256]
        input_image = torch.cat([input_image, rays_embeddings], dim=2) # [1, 4, 9, H, W]
        gaussians = self.model.forward_gaussians(input_image)
        out = self.model.gs.render(gaussians, cam_view, cam_view_proj, cam_pos, bg_color=bg_color) # (B, V, H, W, 3), [0, 1] 
        return out

def load_LGM(device):
    
    # to match zero123plus
    opt.bg = 0.5
    opt.output_size = 320
    model = LGM(opt)

    # resume pretrained checkpoint
    if opt.resume is not None:
        if opt.resume.endswith('safetensors'):
            ckpt = load_file(opt.resume, device='cpu')
        else:
            ckpt = torch.load(opt.resume, map_location='cpu')
        model.load_state_dict(ckpt, strict=False)
        print(f'[INFO] Loaded checkpoint from {opt.resume}')
    else:
        print(f'[WARN] model randomly initialized, are you sure?')

    # device
    model.half().to(device)
    model.eval().requires_grad_(False)

    tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
    proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=device)
    proj_matrix[0, 0] = 1 / tan_half_fov
    proj_matrix[1, 1] = 1 / tan_half_fov
    proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
    proj_matrix[2, 3] = 1
    proj_matrix = proj_matrix
    return model, proj_matrix, opt

def get_cam_poses(elevations, azimuths, opt, device, batch_gpu):
    cam_poses = [orbit_camera(ele, azi, radius=opt.cam_radius, opengl=True) for (ele, azi) in zip(elevations, azimuths)]
    cam_poses = torch.from_numpy(np.stack(cam_poses, axis=0)).to(device)
    cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
    
    cam_view = torch.inverse(cam_poses).transpose(1, 2).unsqueeze(0).repeat(batch_gpu,1,1,1) # [B, V, 4, 4]
    cam_view_proj = cam_view @ proj_matrix # [B, V, 4, 4]
    cam_pos = - cam_poses[:, :3, 3].unsqueeze(0).repeat(batch_gpu,1,1) # [B, V, 3]
    return cam_view, cam_view_proj, cam_pos