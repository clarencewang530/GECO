import numpy as np
from PIL import Image
import torch
import os
import glob
from torchvision import transforms as T
from torch.utils.data import Dataset
from utils.saving import to_rgb_image
import json

class ObjaverseSingleViewData(Dataset):
    def __init__(self, path, source_size=256):
        self.root_dir = path
        self.paths = sorted(glob.glob(os.path.join(self.root_dir, '**/**.png')))
        print('num of images', len(self.paths))
        self.source_size = source_size
    
    @staticmethod
    def _default_intrinsics():
        # return: (3, 2)
        fx = fy = 384
        cx = cy = 256
        w = h = 512
        intrinsics = torch.tensor([
            [fx, fy],
            [cx, cy],
            [w, h],
        ], dtype=torch.float32)
        return intrinsics

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # return img and pose
        filename = self.paths[index]
        img = np.array(to_rgb_image(Image.open(filename).resize((self.source_size, self.source_size))))
        return {
            'img': img
        }

class ObjaverseZero123Data(Dataset):
    def __init__(self, path, source_size=256):
        self.root_dir = path
        self.paths = json.load(open(os.path.join(path, 'valid_paths.json')))
        self.source_size = source_size
    
    @staticmethod
    def _default_intrinsics():
        # return: (3, 2)
        fx = fy = 384
        cx = cy = 256
        w = h = 512
        intrinsics = torch.tensor([
            [fx, fy],
            [cx, cy],
            [w, h],
        ], dtype=torch.float32)
        return intrinsics

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        # return img and pose
        # filename = self.paths[index]
        filename = os.path.join(self.root_dir, self.paths[index], '000.png')
        img = np.array(to_rgb_image(Image.open(filename).resize((self.source_size, self.source_size))))
        return {
            'img': img
        }

class NoiseImagePairDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.all_img_path = sorted(glob.glob(os.path.join(self.path, '*', '6view.png')))
        self.all_cond_path = sorted(glob.glob(os.path.join(self.path, '*', 'cond.png')))
        self.all_noise_path = sorted(glob.glob(os.path.join(self.path, '*', '6view-z.npy')))
        self.all_latent_path = sorted(glob.glob(os.path.join(self.path, '*', '6view-latents.npy')))
    
    def __len__(self):
        return len(self.all_img_path)

    def __getitem__(self, idx):
        z = torch.from_numpy(np.load(self.all_noise_path[idx])).squeeze().float()
        latent = torch.from_numpy(np.load(self.all_latent_path[idx])).squeeze().float()
        image = Image.open(self.all_img_path[idx])
        image = T.ToTensor()(image) * 2.0 - 1.0
        cond = np.array(Image.open(self.all_cond_path[idx]))

        return {
            'cond': cond,
            'image': image,
            'latent': latent,
            'z': z
        }