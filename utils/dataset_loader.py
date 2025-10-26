import os
import torch
import numpy as np
from torch.utils.data import Dataset
import tifffile as tiff

class DualInputGradientDataset(Dataset):
    def __init__(self, boundary_dir, gradient_input_dir, mask_dir, gt_gradient_dir, transform=None):
        self.boundary_paths = sorted([os.path.join(boundary_dir, f) for f in os.listdir(boundary_dir)])
        self.gradient_input_paths = sorted([os.path.join(gradient_input_dir, f) for f in os.listdir(gradient_input_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        self.gt_gradient_paths = sorted([os.path.join(gt_gradient_dir, f) for f in os.listdir(gt_gradient_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.boundary_paths)

    def load_image(self, path):
        if path.endswith('.npy'):
            return np.load(path)
        elif path.endswith('.tif') or path.endswith('.tiff'):
            img = tiff.imread(path).astype(np.float32)
            # Normalize 16-bit TIFF to [0, 1]
            if img.max() > 1:
                img /= 65535.0
            return img
        else:
            raise ValueError(f"Unsupported file type: {path}")

    def __getitem__(self, idx):
        boundary = self.load_image(self.boundary_paths[idx])
        gradient_input = self.load_image(self.gradient_input_paths[idx])
        mask = self.load_image(self.mask_paths[idx])
        gt_gradient = self.load_image(self.gt_gradient_paths[idx])  # already 0-1

        mask = (mask > 0).astype(np.float32)

        sample = {
            'input1': torch.tensor(boundary, dtype=torch.float32).unsqueeze(0),
            'input2': torch.tensor(gradient_input, dtype=torch.float32).unsqueeze(0),
            'mask': torch.tensor(mask, dtype=torch.float32).unsqueeze(0),
            'gradient': torch.tensor(gt_gradient, dtype=torch.float32).unsqueeze(0)
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
