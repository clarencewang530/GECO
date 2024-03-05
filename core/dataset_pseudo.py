import os
import cv2
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

import kiui
from core.options import Options
from core.utils import get_rays, grid_distortion, orbit_camera_jitter
import json
from PIL import Image
import einops

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)


class NoiseImageLGMDataset(Dataset):
    def __init__(self, opt: Options, training=True):
        self.path = opt.data_path
        self.scenes = sorted(os.listdir(self.path))
        import warnings
        warnings.warn('not using full dataset yet')
        self.opt = opt
        self.training = training

        if self.training:
            self.scenes = self.scenes[:-self.opt.batch_size]
        else:
            self.scenes = self.scenes[-self.opt.batch_size:]

        self.tan_half_fov = np.tan(0.5 * np.deg2rad(self.opt.fovy))
        self.proj_matrix = torch.zeros(4, 4, dtype=torch.float32)
        self.proj_matrix[0, 0] = 1 / self.tan_half_fov
        self.proj_matrix[1, 1] = 1 / self.tan_half_fov
        self.proj_matrix[2, 2] = (self.opt.zfar + self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[3, 2] = - (self.opt.zfar * self.opt.znear) / (self.opt.zfar - self.opt.znear)
        self.proj_matrix[2, 3] = 1
    
    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, idx):
        results = {}
        cam = json.load(open(os.path.join(self.path, self.scenes[idx], 'cam.json')))
        elevations, azimuths = np.array(cam['elevation']), np.array(cam['azimuth'])
        vids = np.arange(6).tolist() + np.random.permutation(np.arange(6, 50)).tolist()

        z = torch.from_numpy(np.load(os.path.join(self.path, self.scenes[idx], 'z.npy')))
        cond = np.array(Image.open( os.path.join(self.path, self.scenes[idx], 'cond.png') ))

        six_view_path = os.path.join(self.path, self.scenes[idx], '6view.png')
        six_view = np.array(Image.open(six_view_path).resize((512, 768))) / 255.0
        six_view = einops.rearrange(six_view, '(h2 h) (w2 w) c -> (h2 w2) c h w', h2=3, w2=2).astype(np.float32) # (6, 320, 320, 3)
        images_input = torch.from_numpy(six_view)

        # onestep_path = os.path.join(self.path, self.scenes[idx], 'onestep.png')
        # onestep_view = np.array(Image.open(onestep_path).resize((512, 768))) / 255.0
        # onestep_view = einops.rearrange(onestep_view, '(h2 h) (w2 w) c -> (h2 w2) c h w', h2=3, w2=2).astype(np.float32) # (6, 320, 320, 3)
        # images_input = torch.from_numpy(onestep_view)

        images = []
        masks = []
        cam_poses = []
        for i, vid in enumerate(vids[:self.opt.num_views]):
            image_path = os.path.join(self.path, self.scenes[idx], f'{vid:03d}.png')
            image = np.array(Image.open(image_path)) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1).float()
            mask_path = os.path.join(self.path, self.scenes[idx], f'{vid:03d}-mask.png')
            mask = torch.from_numpy(np.array(Image.open(mask_path)) / 255.0).to(image.dtype)

            from kiui.cam import orbit_camera
            c2w = orbit_camera(elevations[vid], azimuths[vid], radius=1.5)
            c2w = torch.from_numpy(c2w)
            c2w[:3, 3] *= self.opt.cam_radius / 1.5

            images.append(image)
            masks.append(mask)
            cam_poses.append(c2w)
        
        images = torch.stack(images, dim=0) # [V, C, H, W]
        masks = torch.stack(masks, dim=0) # [V, H, W]
        cam_poses = torch.stack(cam_poses, dim=0) # [V, 4, 4]

        transform = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, self.opt.cam_radius], [0, 0, 0, 1]], dtype=torch.float32) @ torch.inverse(cam_poses[0])
        cam_poses = transform.unsqueeze(0) @ cam_poses  # [V, 4, 4]

        images_input = F.interpolate(images_input, size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False) # [V, C, H, W]
        cam_poses_input = cam_poses[:self.opt.num_input_views].clone()
        images_input = TF.normalize(images_input, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)

        # resize render ground-truth images, range still in [0, 1]
        results['images_output'] = F.interpolate(images, size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, C, output_size, output_size]
        results['masks_output'] = F.interpolate(masks.unsqueeze(1), size=(self.opt.output_size, self.opt.output_size), mode='bilinear', align_corners=False) # [V, 1, output_size, output_size]

        rays_embeddings = []
        for i in range(self.opt.num_input_views):
            rays_o, rays_d = get_rays(cam_poses_input[i], self.opt.input_size, self.opt.input_size, self.opt.fovy) # [h, w, 3]
            rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
            rays_embeddings.append(rays_plucker)

        rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous() # [V, 6, h, w]
        final_input = torch.cat([images_input, rays_embeddings], dim=1) # [V=4, 9, H, W]
        results['input'] = final_input
        results['rays_embeddings'] = rays_embeddings

        # opengl to colmap camera for gaussian renderer
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        # cameras needed by gaussian rasterizer
        cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3] # [V, 3]
        
        results['cam_view'] = cam_view
        results['cam_view_proj'] = cam_view_proj
        results['cam_pos'] = cam_pos

        results['cond'] = cond
        results['z'] = z
        results['index'] = self.scenes[idx]

        return results