import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import kiui
from kiui.lpips import LPIPS
from diffusers import DiffusionPipeline

from core.options import Options

class Zero123PlusNPlane(Zero123PlusGaussian):
    def __init__(
        self,
        opt: Options,
    ):
        super().__init__()
        self.opt = opt

        # Load zero123plus model
        self.pipe = DiffusionPipeline.from_config(
            opt.model_path,
            custom_pipeline=opt.custom_pipeline
        )
        self.vae = self.pipe.vae.eval()
        self.unet = self.pipe.unet.train().requires_grad_(True)

        # TODO: add auxiliary layer for NPlane

        # Gaussian Renderer
        # LPIPS loss
        if self.opt.lambda_lpips > 0:
            self.lpips_loss = LPIPS(net='vgg')
            self.lpips_loss.requires_grad_(False)
    
    def forward(self, data, step_ratio=1.0):
        pass