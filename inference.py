import os
import torch
import numpy as np
import tifffile
from tqdm import tqdm
from utils.model import DualEncoderUNet

# === Load image and normalize to 0-1 for model input ===
def load_image_and_normalize(path):
    img = tifffile.imread(path).astype(np.float32)
    if img.ndim > 2:
        img = img.squeeze()
    # Scale 16-bit image to 0–1 (matches training preprocessing)
    img /= 255.0
    return img

# === Inference function ===
def run_inference_scale_back(input1_dir, input2_dir, output_dir, model, device='cuda'):
    model.to(device)
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    input1_files = sorted(os.listdir(input1_dir))
    input2_files = sorted(os.listdir(input2_dir))

    assert len(input1_files) == len(input2_files), "Mismatched input folders"

    for fname1, fname2 in tqdm(zip(input1_files, input2_files), total=len(input1_files), desc="Running inference"):
        name = os.path.splitext(fname1)[0]

        # Load and normalize
        img1 = load_image_and_normalize(os.path.join(input1_dir, fname1))
        img2 = load_image_and_normalize(os.path.join(input2_dir, fname2))

        # Convert to tensor and add batch & channel dims
        img1_tensor = torch.tensor(img1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
        img2_tensor = torch.tensor(img2, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = model(img1_tensor, img2_tensor)

        pred = pred.squeeze().cpu().numpy()

        # Zero-out negatives (optional)
        pred_thresh = np.where(pred > -0.5, pred, 0.0)

        # Scale back to 16-bit range (0–65535)
        pred_16bit = np.clip(pred_thresh * 65535.0, 0, 65535).astype(np.uint16)

        # Save as 16-bit TIFF
        out_path = os.path.join(output_dir, f"{name}_ficm.tif")
        tifffile.imwrite(out_path, pred_16bit)

    print("✅ Inference complete. Results saved in:", output_dir)

# === Example usage ===
if __name__ == "__main__":
    model = DualEncoderUNet()
    model.load_state_dict(torch.load("utils/saved_model/wholeslide.pth", map_location='cpu'))

    run_inference_scale_back(
        input1_dir = "project_1_training_data/HCC827_TEST_DATA/fluorescence_image",
        input2_dir = "project_1_training_data/HCC827_TEST_DATA/grid_image",
        output_dir = "OUTPUT",
        model=model,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
