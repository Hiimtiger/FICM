import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utils.model import SimpleCNN
from utils.dataset import get_train_val_loaders
import os

def train_model(train_dir, model_name, epochs=100, batch_size=4, lr=0.001):
    # Get train & validation loaders
    train_loader, val_loader = get_train_val_loaders(train_dir, batch_size=batch_size, seed=42)

    # --- Add label distribution check here ---
    from collections import Counter
    all_train_labels = []
    for _, labels in train_loader:
        all_train_labels.extend(labels.tolist())  # flatten all batch labels into a list

    print("Training set label distribution:", Counter(all_train_labels))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN().to(device)
    model.fc[-1] = nn.Identity()  # Remove sigmoid for BCEWithLogitsLoss

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Track best validation loss
    best_val_loss = float("inf")

    save_dir = "utils/saved_model"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}.pth")

    for epoch in range(epochs):
        # ----- Training -----
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            train_total += labels.size(0)
            train_correct += (preds == labels).sum().item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # ----- Validation -----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                val_total += labels.size(0)
                val_correct += (preds == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {avg_train_loss:.4f} "
              f"Train Acc: {100 * train_correct/train_total:.2f}% | "
              f"Val Loss: {avg_val_loss:.4f} "
              f"Val Acc: {100 * val_correct/val_total:.2f}%")

        # ----- Save best model -----
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"✅ New best model saved! (Val Loss: {best_val_loss:.4f})")
        else:
            print("No improvement, model not saved.")

    print(f"Training finished. Best model saved at: {save_path}")
