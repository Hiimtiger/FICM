import os
import re
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image

class ZScoreNormalize(object):
    def __call__(self, tensor):
        mean = tensor.mean()
        std = tensor.std()
        if std > 0:
            return (tensor - mean) / std
        else:
            return tensor - mean

class CellDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.file_list = [f for f in os.listdir(image_dir) if f.lower().endswith(".tif")]
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def get_label_from_name(self, filename):
        match = re.match(r"^(A|HCC)_\d+\.tif$", filename, re.IGNORECASE)
        if match:
            label_str = match.group(1).upper()
            return 1 if label_str == "HCC" else 0
        else:
            raise ValueError(f"Filename {filename} does not match expected pattern.")

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.image_dir, img_name)

        image = Image.open(img_path).convert("L")
        label = self.get_label_from_name(img_name)
        label = torch.tensor(label, dtype=torch.float32)  # use float32 for BCEWithLogitsLoss

        if self.transform:
            image = self.transform(image)

        return image, label

def get_train_val_loaders(image_dir, batch_size=16, val_ratio=0.2, seed=42):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        ZScoreNormalize(),
    ])

    dataset = CellDataset(image_dir, transform=transform)

    total_size = len(dataset)
    val_size = int(val_ratio * total_size)
    train_size = total_size - val_size

    # Shuffle indices before splitting
    torch.manual_seed(seed)
    indices = torch.randperm(total_size)

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
