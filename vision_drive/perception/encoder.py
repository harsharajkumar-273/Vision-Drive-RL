import torch
import torch.nn as nn

class CameraEncoder(nn.Module):
    """
    A CNN encoder that processes raw perspective camera frames.
    Input shape: (batch_size, channels=3, height=64, width=64)
    Output shape: (batch_size, hidden_dim)
    """
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),  # -> 16 x 32 x 32
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # -> 32 x 16 x 16
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> 64 x 8 x 8
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # -> 128 x 4 x 4
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x):
        features = self.conv(x)
        out = self.fc(features)
        return out
