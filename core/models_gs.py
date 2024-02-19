import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import kiui
from kiui.lpips import LPIPS
from diffusers import DiffusionPipeline, DDPMScheduler
from PIL import Image
import einops

from core.options import Options
from core.gs import GaussianRenderer

def unscale_latents(latents):
    latents = latents / 0.75 + 0.22
    return latents

def unscale_image(image):
    image = image / 0.5 * 0.8
    return image

def scale_image(image):
    image = image * 0.5 / 0.8
    return image

def scale_latents(latents):
    latents = (latents - 0.22) * 0.75
    return latents

def encode_image(image, vae, is_zero123plus=True):
    if is_zero123plus:
        image = scale_image(image)
        image = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
        image = scale_latents(image)
    else:
        image = vae.encode(image, return_dict=False)[0] * vae.config.scaling_factor
    return image

def decode_latents(latents, decoder, is_zero123plus=True):
    if is_zero123plus:
        latents = unscale_latents(latents)
        latents = latents / decoder.vae.config.scaling_factor
        # image = decoder.decode(latents, return_dict=False)[0]
        image = decoder(latents)
        image = unscale_image(image)
    else:
        image = decoder.decode(latents, return_dict=False)[0]
    return image

def to_rgb_image(maybe_rgba: Image.Image):
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = np.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)

def predict_noise0_diffuser(unet, noisy_latents, text_embeddings, t, 
    guidance_scale=1.0, cross_attention_kwargs={}, scheduler=None, lora_v=False, model='sd'):
    batch_size = noisy_latents.shape[0]
    latent_model_input = torch.cat([noisy_latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, t)

    if type(t) == int:
        t = torch.tensor([t] * batch_size, device=noisy_latents.device)
    # https://github.com/threestudio-project/threestudio/blob/77de7d75c34e29a492f2dda498c65d2fd4a767ff/threestudio/models/guidance/stable_diffusion_vsd_guidance.py#L512
    alphas_cumprod = scheduler.alphas_cumprod.to(
        device=noisy_latents.device, dtype=noisy_latents.dtype
    )
    alpha_t = alphas_cumprod[t] ** 0.5
    sigma_t = (1 - alphas_cumprod[t]) ** 0.5
    if guidance_scale == 1.:
        cak = {}
        if cross_attention_kwargs is not None:
            for key in cross_attention_kwargs:
                if isinstance(cross_attention_kwargs[key], torch.Tensor):
                    cak[key] = cross_attention_kwargs[key][batch_size:]
                else:
                    cak[key] = cross_attention_kwargs[key]
        noise_pred = unet(noisy_latents, t, encoder_hidden_states=text_embeddings[batch_size:], cross_attention_kwargs=cak).sample
        if lora_v or scheduler.config.prediction_type == 'v_prediction':
            # assume the output of unet is v-pred, convert to noise-pred now
            noise_pred = noisy_latents * sigma_t.view(-1, 1, 1, 1) + noise_pred * alpha_t.view(-1, 1, 1, 1)
        if model == 'unet_if':
            noise_pred, _ = noise_pred.split(3, dim=1)
    else:
        t = torch.cat([t] * 2)
        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings, cross_attention_kwargs=cross_attention_kwargs).sample
        if lora_v or scheduler.config.prediction_type == 'v_prediction':
            # assume the output of unet is v-pred, convert to noise-pred now
            noise_pred = latent_model_input * torch.cat([sigma_t] * 2, dim=0).view(-1, 1, 1, 1) + noise_pred * torch.cat([alpha_t] * 2, dim=0).view(-1, 1, 1, 1)
        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        if model == 'unet_if':
            noise_pred_text, predicted_variance = noise_pred_text.split(3, dim=1)
            noise_pred_uncond, _ = noise_pred_uncond.split(3, dim=1)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    return noise_pred

def get_alpha(scheduler, t, device):
    alphas_cumprod = scheduler.alphas_cumprod.to(device=device)
    alpha_t = alphas_cumprod[t] ** 0.5
    sigma_t = (1 - alphas_cumprod[t]) ** 0.5
    return alpha_t, sigma_t
    
def predict_x0(unet, noisy_latents, text_embeddings, t, 
        guidance_scale=1.0, cross_attention_kwargs={}, 
        scheduler=None, lora_v=False, model='sd'):
    alpha_t, sigma_t = get_alpha(scheduler, t, noisy_latents.device)
    noise_pred = predict_noise0_diffuser(
        unet, noisy_latents, text_embeddings, t=t,
        guidance_scale=guidance_scale, cross_attention_kwargs=cross_attention_kwargs, 
        scheduler=scheduler, model=model
    )
    return (noisy_latents - noise_pred * sigma_t) / alpha_t

# From: https://github.com/huggingface/diffusers/blob/v0.26.3/src/diffusers/models/autoencoders/autoencoder_kl.py#L35
class UNetDecoder(nn.Module):
    def __init__(self, vae):
        super(UNetDecoder, self).__init__()
        self.vae = vae
        self.decoder = vae.decoder
        self.others = nn.Conv2d(128, 11, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    
    def forward(self, z):
        sample = self.vae.post_quant_conv(z)
        latent_embeds = None
        sample = self.decoder.conv_in(sample)
        upscale_dtype = next(iter(self.decoder.up_blocks.parameters())).dtype
        sample = self.decoder.mid_block(sample, latent_embeds)
        sample = sample.to(upscale_dtype)
        # up
        for up_block in self.decoder.up_blocks:
            sample = up_block(sample, latent_embeds)

        sample = self.decoder.conv_norm_out(sample)
        sample = self.decoder.conv_act(sample)
        rgb = self.decoder.conv_out(sample)
        others = self.others(sample)
        return torch.cat([others, rgb], dim=1)
        # return rgb


class Zero123PlusGaussian(nn.Module):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()
        self.opt = opt

        # Load zero123plus model
        import sys
        sys.path.append('./pipelines')
        self.pipe = DiffusionPipeline.from_pretrained(
            opt.model_path,
            custom_pipeline=opt.custom_pipeline
        ).to('cuda')

        self.pipe.prepare() 
        self.vae = self.pipe.vae.requires_grad_(False).eval()
        self.vae.decoder.requires_grad_(False).eval()

        self.unet = self.pipe.unet.eval().requires_grad_(False)
        self.pipe.scheduler = DDPMScheduler.from_config(self.pipe.scheduler.config)
        self.decoder = UNetDecoder(self.vae)

        # add auxiliary layer for generating gaussian (14 channels)
        # self.conv = nn.Conv2d(3, 14, kernel_size=3, stride=1, padding=1)

        # Gaussian Renderer
        self.gs = GaussianRenderer(opt)
        # activations...
        self.pos_act = lambda x: x.clamp(-1, 1)
        self.scale_act = lambda x: 0.1 * F.softplus(x)
        self.opacity_act = lambda x: torch.sigmoid(x)
        self.rot_act = F.normalize
        self.rgb_act = lambda x: 0.5 * torch.tanh(x) + 0.5 # NOTE: may use sigmoid if train again

        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)

    def state_dict(self, **kwargs):
        # remove lpips_loss
        state_dict = super().state_dict(**kwargs)
        for k in list(state_dict.keys()):
            if 'lpips_loss' in k:
                del state_dict[k]
        return state_dict

    def forward_gaussians(self, images, cond):
        B, V, C, H, W = images.shape
        with torch.no_grad():
            text_embeddings, cross_attention_kwargs = self.pipe.prepare_conditions(cond, guidance_scale=4.0)
            cross_attention_kwargs_stu = cross_attention_kwargs

        # make input 6 views into a 3x2 grid
        images = einops.rearrange(images, 'b (h2 w2) c h w -> b c (h2 h) (w2 w)', h2=3, w2=2) 

        # scale as in zero123plus
        latents = encode_image(images, self.vae)
        t = torch.tensor([10] * B, device=latents.device)
        latents = self.pipe.scheduler.add_noise(latents, torch.randn_like(latents, device=latents.device), t)
        x = predict_x0(
            self.unet, latents, text_embeddings, t=10, guidance_scale=1.0, 
            cross_attention_kwargs=cross_attention_kwargs, scheduler=self.pipe.scheduler, model='zero123plus')
        # x = torch.randn([B, 4, 96, 64], device=images.device)
        x = decode_latents(x, self.decoder) # (B, 14, H, W)
        # x = self.conv(x)

        x = einops.rearrange(x, 'b c (h2 h) (w2 w) -> b (h2 w2) c h w', h2=3, w2=2) # (B*6, 14, H, W)
        x = x.reshape(B*6, -1, H, W)

        x = x.reshape(B, V, 14, 256, 256)
        x = x.permute(0, 1, 3, 4, 2).reshape(B, -1, 14)
        
        pos = self.pos_act(x[..., :3]) # [B, N, 3]
        opacity = self.opacity_act(x[..., 3:4])
        scale = self.scale_act(x[..., 4:7])
        rotation = self.rot_act(x[..., 7:11])
        rgbs = x[..., 11:] # FIXME: original activation removed

        gaussians = torch.cat([pos, opacity, scale, rotation, rgbs], dim=-1) # [B, N, 14]
        return gaussians
    
    def forward(self, data, step_ratio=1):
        # Gaussian shape: (B*6, 14, H, W)
        results = {}
        loss = 0

        images = data['input'] # [B, 4, 9, h, W], input features
        cond = data['cond'] # [B, H, W, 3], condition image
        
        # use the first view to predict gaussians
        gaussians = self.forward_gaussians(images, cond) # [B, N, 14]

        results['gaussians'] = gaussians

        # random bg for training
        if self.training:
            bg_color = torch.rand(3, dtype=torch.float32, device=gaussians.device)
        else:
            bg_color = torch.ones(3, dtype=torch.float32, device=gaussians.device) * 0.5

        # use the other views for rendering and supervision
        results = self.gs.render(gaussians, data['cam_view'], data['cam_view_proj'], data['cam_pos'], bg_color=bg_color)
        pred_images = results['image'] # [B, V, C, output_size, output_size]
        pred_alphas = results['alpha'] # [B, V, 1, output_size, output_size]

        results['images_pred'] = pred_images
        results['alphas_pred'] = pred_alphas

        gt_images = data['images_output'] # [B, V, 3, output_size, output_size], ground-truth novel views
        gt_masks = data['masks_output'] # [B, V, 1, output_size, output_size], ground-truth masks

        gt_images = gt_images * gt_masks + bg_color.view(1, 1, 3, 1, 1) * (1 - gt_masks)

        loss_mse = F.mse_loss(pred_images, gt_images) + F.mse_loss(pred_alphas, gt_masks)
        loss = loss + loss_mse

        if self.opt.lambda_lpips > 0:
            loss_lpips = self.lpips_loss(
                # gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1,
                # downsampled to at most 256 to reduce memory cost
                F.interpolate(gt_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False), 
                F.interpolate(pred_images.view(-1, 3, self.opt.output_size, self.opt.output_size) * 2 - 1, (256, 256), mode='bilinear', align_corners=False),
            ).mean()
            results['loss_lpips'] = loss_lpips
            loss = loss + self.opt.lambda_lpips * loss_lpips
            
        results['loss'] = loss

        # metric
        with torch.no_grad():
            psnr = -10 * torch.log10(torch.mean((pred_images.detach() - gt_images) ** 2))
            results['psnr'] = psnr

        return results
    