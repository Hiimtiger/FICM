import torch
import torch.nn as nn
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

def dice_loss(pred, target, smooth=1e-6):
    """
    Compute Dice loss for binary masks.
    """
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (pred * target).sum(dim=1)
    dice = (2. * intersection + smooth) / (pred.sum(dim=1) + target.sum(dim=1) + smooth)
    return 1 - dice.mean()

def smooth_l1_loss(pred, target):
    """
    Standard Smooth L1 loss over the entire prediction map.
    """
    loss_fn = nn.SmoothL1Loss()
    return loss_fn(pred, target)

def save_sample_images(epoch, input1, input2, masks, gt_gradients, outputs):
    import numpy as np
    fold_dir = os.path.join("utils/sample_image")
    os.makedirs(fold_dir, exist_ok=True)

    for i in range(min(3, len(input1))):
        # Prepare input1 as red channel
        red = input1[i].cpu().numpy().squeeze()
        red = (red - red.min()) / (red.max() - red.min() + 1e-8)

        # Prepare input2 as grayscale
        gray = input2[i].cpu().numpy().squeeze()
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)

        # Compose RGB image
        rgb_img = np.stack([red, gray, gray], axis=-1)

        # Prepare mask, ground truth gradient, and predicted output
        mask = masks[i].cpu().numpy().squeeze()
        gt_grad = gt_gradients[i].cpu().numpy().squeeze()
        output = outputs[i].cpu().numpy().squeeze()

        # Plotting
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 5))

        axes[0].imshow(rgb_img)
        axes[0].set_title("Input")

        axes[1].imshow(gt_grad)
        axes[1].set_title("Ground Truth Gradient")

        axes[2].imshow(output)
        axes[2].set_title("Predicted Heatmap")

        for ax in axes:
            ax.axis("off")

        plt.tight_layout()
        plt.savefig(f"{fold_dir}/epoch{epoch + 1}_sample{i + 1}.png")
        plt.close()

def train(model, train_loader, val_loader, optimizer, num_epochs, device='cuda', save_path='utils/saved_model/best_model.pth'):
    model = model.to(device)
    best_val_loss = float('inf')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_dice = 0.0
        total_l1 = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for batch in pbar:
            input1 = batch['input1'].to(device)
            input2 = batch['input2'].to(device)
            gt_mask = batch['mask'].to(device)
            gt_grad = batch['gradient'].to(device)

            optimizer.zero_grad()
            output = model(input1, input2)

            pred_mask = (output > 0.05).float()
            loss_dice = dice_loss(pred_mask, gt_mask)
            loss_grad = smooth_l1_loss(output, gt_grad)

            loss = loss_dice + loss_grad
            loss.backward()
            optimizer.step()

            total_dice += loss_dice.item()
            total_l1 += loss_grad.item()
            pbar.set_postfix({'Dice': loss_dice.item(), 'SmoothL1': loss_grad.item()})

        avg_train_loss = (total_dice + total_l1) / len(train_loader)
        print(f"[Epoch {epoch+1}] Train Dice: {total_dice/len(train_loader):.4f}, Train SmoothL1: {total_l1/len(train_loader):.4f}")

        # === Validation ===
        model.eval()
        val_dice = 0.0
        val_l1 = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                input1 = batch['input1'].to(device)
                input2 = batch['input2'].to(device)
                gt_mask = batch['mask'].to(device)
                gt_grad = batch['gradient'].to(device)

                output = model(input1, input2)
                pred_mask = (output > 0.05).float()
                loss_dice = dice_loss(pred_mask, gt_mask)
                loss_grad = smooth_l1_loss(output, gt_grad)

                val_dice += loss_dice.item()
                val_l1 += loss_grad.item()

            if (epoch + 1) % 5 == 0:
                # Take the first batch from validation loader
                save_sample_images(epoch, input1, input2, gt_mask, gt_grad, output)

        avg_val_loss = (val_dice + val_l1) / len(val_loader)
        print(f"[Epoch {epoch+1}] Val Dice: {val_dice/len(val_loader):.4f}, Val SmoothL1: {val_l1/len(val_loader):.4f}")

        # === Save best model ===
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)