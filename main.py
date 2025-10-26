from utils.model import DualEncoderUNet
from torch.utils.data import DataLoader, random_split
from utils.dataset_loader import DualInputGradientDataset
from utils.trainer import train

import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    
    set_seed(42)  

    # === Paths ===
    boundary_dir = 'fluorescence'
    gradient_input_dir = 'force_grid_images'
    mask_dir = 'gt_mask'
    gt_gradient_dir = 'gt_gradients'
    save_path = 'utils/saved_model/wholeslide.pth'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # === Hyperparameters ===
    batch_size = 4
    num_epochs = 150
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Dataset ===
    full_dataset = DualInputGradientDataset(boundary_dir, gradient_input_dir, mask_dir, gt_gradient_dir)

    # === 80/20 Split with seeded generator ===
    val_ratio = 0.2
    total_size = len(full_dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size

    generator = torch.Generator().manual_seed(42)  # Seed generator for reproducible split
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # === Model ===
    model = DualEncoderUNet(in_ch1=1, in_ch2=1, out_ch=1).to(device)

    # === Optimizer ===
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # === Training ===
    train(model, train_loader, val_loader, optimizer, num_epochs=num_epochs, device=device, save_path=save_path)

if __name__ == '__main__':
    
    main()
