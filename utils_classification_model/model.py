import torch
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1),  # Conv1 -> 256x256
            nn.ReLU(),
            nn.MaxPool2d(2),  # 128x128

            nn.Conv2d(16, 32, 3, padding=1),  # Conv2 -> 128x128
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(32, 64, 3, padding=1),  # Conv3 -> 64x64
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(64, 128, 3, padding=1),  # Conv4 -> 32x32
            nn.ReLU(),
            nn.MaxPool2d(2)  # 16x16
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Binary classification output
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc(x)
        return x 