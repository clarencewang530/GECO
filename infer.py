
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

from core.options import AllConfigs, Options
from core.models_gs import Zero123PlusGaussian
from core.models_lgm import LGM
from core.models_single import SingleSplatterImage
from core.models_gs import predict_x0, decode_latents
from pipelines.pipeline_mvdream import MVDreamPipeline
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler, DDPMScheduler
from PIL import Image
import einops
import pickle
import time
import json
import random

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.parameters()) + list(module.buffers())

def named_params_and_buffers(module):
    assert isinstance(module, torch.nn.Module)
    return list(module.named_parameters()) + list(module.named_buffers())

@torch.no_grad()
def copy_params_and_buffers(src_module, dst_module, require_all=False):
    assert isinstance(src_module, torch.nn.Module)
    assert isinstance(dst_module, torch.nn.Module)
    src_tensors = dict(named_params_and_buffers(src_module))
    for name, tensor in named_params_and_buffers(dst_module):
        assert (name in src_tensors) or (not require_all)
        if name in src_tensors:
            tensor.copy_(src_tensors[name].detach()).requires_grad_(tensor.requires_grad)

def default_rays_from_pose(device, cam_poses, opt):
    from core.utils import get_rays
    rays_embeddings = []
    for i in range(cam_poses.shape[0]):
        rays_o, rays_d = get_rays(cam_poses[i], opt.input_size, opt.input_size, opt.fovy) # [h, w, 3]
        rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d], dim=-1) # [h, w, 6]
        rays_embeddings.append(rays_plucker)
    rays_embeddings = torch.stack(rays_embeddings, dim=0).permute(0, 3, 1, 2).contiguous().to(device) # [V, 6, h, w]
    return rays_embeddings

class Inferrer:
    def __init__(self, opt, device='cuda'):
        self.opt = opt
        # model
        if opt.model_type == 'Zero123PlusGaussian':
            self.model = Zero123PlusGaussian(opt)
        elif opt.model_type == 'LGM':
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
        self.device = torch.device(device)
        self.model.half().to(self.device)
        self.model.eval()

        tan_half_fov = np.tan(0.5 * np.deg2rad(opt.fovy))
        proj_matrix = torch.zeros(4, 4, dtype=torch.float32, device=self.device)
        proj_matrix[0, 0] = 1 / tan_half_fov
        proj_matrix[1, 1] = 1 / tan_half_fov
        proj_matrix[2, 2] = (opt.zfar + opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[3, 2] = - (opt.zfar * opt.znear) / (opt.zfar - opt.znear)
        proj_matrix[2, 3] = 1
        self.proj_matrix = proj_matrix
        self.pipe = self.load_pipeline(opt.pipeline)
    
    def load_pipeline(self, pipeline):
        if pipeline == 'mvdream':
            pipe = MVDreamPipeline.from_pretrained(
                "ashawkey/imagedream-ipmv-diffusers", # remote weights
                torch_dtype=torch.float16,
                trust_remote_code=True,
                # local_files_only=True,
            ).to(self.device)
        elif pipeline.startswith('zero123plus'):
            import sys
            sys.path.append('./pipelines/')
            pipe = DiffusionPipeline.from_pretrained(
                "sudo-ai/zero123plus-v1.1",
                custom_pipeline="./pipelines/zero123plus.py",
                torch_dtype=torch.float16,
            ).to(self.device)
            pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
                pipe.scheduler.config, timestep_spacing='trailing'
            )
            if pipeline == 'zero123plus1step':
                pipe.prepare()
                pipe.scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
                # resume_pkl = "/mnt/kostas-graid/sw/envs/chenwang/workspace/instant123-old/training-runs/zero123plus/zero123plus-gpus1-batch1-same-vsd-20240221-060250-cond0/network-snapshot-005000.pkl"
                # resume_pkl = '/mnt/kostas-graid/sw/envs/chenwang/workspace/instant123-old/training-runs/zero123plus/zero123plus-gpus1-batch1-same-vsd-20240221-064415-cond500/network-snapshot-020000.pkl'
                # resume_pkl='/mnt/kostas-graid/sw/envs/chenwang/workspace/instant123-old/training-runs/zero123plus/zero123plus-gpus1-batch1-same-vsd-20240221-060105-cond200/network-snapshot-005000.pkl'
                # resume_pkl = '/mnt/kostas-graid/sw/envs/chenwang/workspace/instant123-old/training-runs/zero123plus/zero123plus-gpus1-batch1-same-vsd-20240221-060250-cond0/network-snapshot-015000.pkl'

                # resume_pkl='/mnt/kostas-graid/sw/envs/chenwang/workspace/instant123-old/training-runs/zero123plus/zero123plus-gpus1-batch1-same-vsd-20240221-060105-cond200/network-snapshot-030000.pkl'
                # resume_pkl = '/mnt/kostas-graid/sw/envs/chenwang/workspace/instant123-old/training-runs/zero123plus/zero123plus-gpus1-batch1-same-vsd-20240223-222021-cond0_t950/network-snapshot-020000.pkl'
                # resume_pkl = "/mnt/kostas-graid/sw/envs/chenwang/workspace/instant123-old/training-runs/zero123plus/zero123plus-gpus1-batch1-same-vsd-20240224-060403-cond500_t950/network-snapshot-005000.pkl"
                resume_pkl = "/mnt/kostas-graid/sw/envs/chenwang/workspace/instant123-old/training-runs1/zero123plus-lvis/zero123plus-gpus1-batch1-same-vsd-20240301-192604-cond0_t950_lvis/network-snapshot-005000.pkl"
                resume_data = pickle.load(open(resume_pkl, 'rb'))
                copy_params_and_buffers(resume_data['G'], pipe.unet, require_all=False)
                pipe.unet.eval()
                pipe.unet.is_generator = True
        return pipe
 
    def generate_mv_image(self, image):
        t1 = time.time()
        if self.opt.pipeline == 'mvdream':
            # expect input to be (0, 1)
            mv_image = self.pipe('', image.astype(np.float32) / 255.0, guidance_scale=5.0, num_inference_steps=30, elevation=0)
            mv_image = np.stack([mv_image[1], mv_image[2], mv_image[3], mv_image[0]], axis=0) # [4, 256, 256, 3], float32, (0, 1)
        elif self.opt.pipeline.startswith('zero123plus'):
            if self.opt.pipeline == 'zero123plus':
                mv_image = self.pipe(Image.fromarray(image.astype(np.uint8)), num_inference_steps=75).images[0]
            else:
                text_embeddings, cross_attention_kwargs = self.pipe.prepare_conditions(image.astype(np.uint8), guidance_scale=4.0)
                cross_attention_kwargs_stu = cross_attention_kwargs
                print(f'preparing time: {time.time() - t1:.2f}s')
                with torch.no_grad():
                    out = predict_x0(self.pipe.unet, torch.randn([1, 4, 120, 80], dtype=text_embeddings.dtype, device=self.device), text_embeddings, t=self.opt.init_t, guidance_scale=1.0, cross_attention_kwargs=cross_attention_kwargs, scheduler=self.pipe.scheduler, model='zero123plus')
                    out = (decode_latents(out, self.pipe.vae, True)[0] + 1)*127.5 # (-1, 1) -> (0, 255)
                mv_image = Image.fromarray(out.float().permute(1, 2, 0).detach().clip(0, 255).cpu().numpy().astype(np.uint8))
            mv_image = np.array(mv_image.resize((512, 768))) # TODO: why 512x768 cannot work directly
            mv_image = einops.rearrange(mv_image, '(h2 h) (w2 w) c -> (h2 w2) h w c', h2=3, w2=2).astype(np.uint8) # [6, 256, 256, 3]
            mv_image = mv_image.astype(np.float32) / 255.0
        print(f'generation time: {time.time() - t1:.2f}s')
        return mv_image
    
    def render_video(self, elevations, azimuths, gaussians):
        images = []
        for (i, (ele, azi)) in enumerate(zip(elevations, azimuths)):
            cam_poses = torch.from_numpy(orbit_camera(ele, azi, radius=self.opt.cam_radius, opengl=True)).unsqueeze(0).to(self.device)
            cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
            
            # cameras needed by gaussian rasterizer
            cam_view = torch.inverse(cam_poses).transpose(1, 2) # [V, 4, 4]
            cam_view_proj = cam_view @ self.proj_matrix # [V, 4, 4]
            cam_pos = - cam_poses[:, :3, 3] # [V, 3]

            image = self.model.gs.render(gaussians, cam_view.unsqueeze(0), cam_view_proj.unsqueeze(0), cam_pos.unsqueeze(0), scale_modifier=1)['image']
            out = (image.squeeze(1).permute(0,2,3,1).contiguous().float().cpu().numpy() * 255).astype(np.uint8)
            images.append(out)
            # kiui.write_image(f'{self.opt.workspace}/{i+6:03d}.png', out[0])
        return np.concatenate(images, axis=0)
    
    def infer(self, cond, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        t1 = time.time()

        if self.opt.pipeline.startswith('zero123plus'):
            elevations, azimuths = [-30, 20, -30, 20, -30, 20], [30, 90, 150, 210, 270, 330]
        elif self.opt.pipeline == 'mvdream':
            elevations, azimuths = [0, 0, 0, 0], [0, 90, 180, 270]
        if self.opt.include_input:
            elevations = [0] + elevations
            azimuths = [0] + azimuths
        cams = [orbit_camera(ele, azi, radius=self.opt.cam_radius) for (ele, azi) in zip(elevations, azimuths)]
        cam_poses = torch.from_numpy(np.stack(cams, axis=0))

        mv_image = self.generate_mv_image(cond)
        kiui.write_image(os.path.join(out_dir, 'mv_image.png'), mv_image.transpose(1, 0, 2, 3).reshape(-1, mv_image.shape[1]*mv_image.shape[0], 3))
        input_image = torch.from_numpy(mv_image).permute(0, 3, 1, 2).float().to(self.device) # [4, 3, 256, 256]
        input_image = F.interpolate(input_image, size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False)
        if opt.model_type == 'LGM':
            if opt.pipeline.startswith('zero123plus'):
                # rays_embeddings = model.prepare_default_rays(device, elevations, azimuths)
                rays_embeddings = default_rays_from_pose(self.device, cam_poses, self.opt)
            else:
                rays_embeddings = self.model.prepare_default_rays(self.device, [0, 0, 0, 0], [0, 90, 180, 270])
            input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            input_image = torch.cat([input_image, rays_embeddings], dim=1).unsqueeze(0) # [1, 4, 9, H, W]
        
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):    
                gaussians = self.model.forward_gaussians(input_image) if self.opt.model_type == 'LGM' else model.forward_gaussians(input_image.unsqueeze(0), cond.astype(np.uint8))
                print('all time', time.time() - t1)
                ## saving gaussians and video
                self.model.gs.save_ply(gaussians, os.path.join(out_dir, 'output.ply'))
                images = self.render_video(np.arange(0, 16, dtype=np.int32) * 0, np.rad2deg(np.arange(16)/16*2*np.pi), gaussians)
                for (i, img) in enumerate(images):
                    kiui.write_image(f'{out_dir}/{i:03d}.png', img)
                imageio.mimwrite(os.path.join(out_dir, 'output.mp4'), images, fps=30)

    def get_cam_poses(self, elevations, azimuths, bs, nv=50):
        cam_poses = [orbit_camera(ele, azi, radius=self.opt.cam_radius, opengl=True) for (ele, azi) in zip(elevations, azimuths)]
        cam_poses = torch.from_numpy(np.stack(cam_poses, axis=0)).to(self.device)
        cam_poses[:, :3, 1:3] *= -1 # invert up & forward direction
        
        cam_view = torch.inverse(cam_poses).transpose(1, 2).reshape(bs, nv, 4, 4) # [B, V, 4, 4]
        cam_view_proj = cam_view @ self.proj_matrix # [B, V, 4, 4]
        cam_pos = - cam_poses[:, :3, 3].repeat(bs, nv, 3) # [B, V, 3]
        return cam_view, cam_view_proj, cam_pos
    
    def process_from_zero123plus(self, zero123out, rays_embeddings, cam_view, cam_view_proj, cam_pos, bg_color=0.5):
        B = zero123out.shape[0]
        input_image = einops.rearrange((zero123out + 1) / 2, 'b c (h2 h) (w2 w) -> b (h2 w2) c h w', h2=3, w2=2).reshape(-1, 3, 320, 320) # (B*V, 3, H, W)
        input_image = F.interpolate(input_image, size=(self.opt.input_size, self.opt.input_size), mode='bilinear', align_corners=False)
        input_image = TF.normalize(input_image, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD).reshape(B, 6, 3, self.opt.input_size, self.opt.input_size) # [1, 4, 3, 256, 256]
        input_image = torch.cat([input_image, rays_embeddings], dim=2) # [1, 4, 9, H, W]
        gaussians = self.model.forward_gaussians(input_image)
        bg_color = torch.ones(3, dtype=input_image.dtype, device=input_image.device) * 0.5
        image = self.model.gs.render(gaussians, cam_view, cam_view_proj, cam_pos, bg_color=bg_color)['image'] # (B, V, H, W, 3), [0, 1] 
        pred_images_lgm = einops.rearrange(image, 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2)
        return pred_images_lgm

if __name__ == "__main__":
    opt = tyro.cli(AllConfigs)
    inferer = Inferrer(opt)

    seed_everything(42)
    # file_paths = json.load(open(opt.test_path, 'r'))
    paths = sorted(os.listdir('/mnt/kostas-graid/sw/envs/chenwang/data/gso/gso_recon_gsec512'))
    file_paths = {f: f'/mnt/kostas-graid/sw/envs/chenwang/data/gso/gso_eval_gsec/{f}/000.png' for f in paths}
    ## testing
    for key in file_paths.keys():
        print(key)
        cond = np.array(Image.open(f'{file_paths[key]}').resize((opt.input_size, opt.input_size)))
        if cond.shape[-1] != 4:
            cond = rembg.remove(cond).astype(np.float32)
            print('removing background, which may take extra time')
        mask = cond[..., 3:4] / 255
        cond_input = cond[..., :3] * mask + (1 - mask) * 255
        cond = cond[..., :3] * mask + (1 - mask) * int(opt.bg * 255)
        # inferer.infer(cond, f'{opt.workspace}/gso_results/{key}/5k-cond{opt.cond_t}-t{opt.init_t}-bf16-{seed}/')
        # inferer.infer(cond, f'/mnt/kostas-graid/sw/envs/chenwang/workspace/gsec_compare/gsec512-500-950-15k-second-stage/{key}')
        inferer.infer(cond, f'./workspace/cond950_0_5k_lvis/{key}')