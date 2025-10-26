import os
import re
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from utils.model import SimpleCNN
import torchvision.transforms as transforms

# --- Simple dataset that reads all images in a folder ---
class AllImagesDataset(Dataset):
    def __init__(self, folder):
        self.image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(".tif")]
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path)
        img = self.transform(img)

        fname = os.path.basename(path)
        if fname.startswith("A_"):
            label = 0
        elif fname.startswith("HCC_"):
            label = 1
        else:
            raise ValueError(f"Unknown label for file: {fname}")

        return img, label

# --- Main testing function ---
def test_model_all(data_dir, model_name="cell_cnn", batch_size=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset & loader
    dataset = AllImagesDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Load model
    model = SimpleCNN().to(device)
    model.fc[-1] = nn.Identity()  # Remove sigmoid for BCEWithLogitsLoss
    model.load_state_dict(torch.load(f"{model_name}.pth", map_location=device))
    model.eval()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            preds = (torch.sigmoid(outputs) >= 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Convert to int lists
    all_labels = [int(x.item()) for x in all_labels]
    all_preds = [int(x.item()) for x in all_preds]


    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix:")
    print(cm)


    # Accuracy
    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return accuracy


