import torch
import torch.nn as nn
from .encoder import CameraEncoder

class BEVProjector(nn.Module):
    """
    Fuses multi-camera features and decodes them into a 2D Bird's-Eye-View (BEV) occupancy map.
    Input: Multi-camera frames (batch_size, 3_cameras, 3_channels, 64_height, 64_width)
    Output: BEV Occupancy Grid (batch_size, 2_channels, 64_height, 64_width)
    """
    def __init__(self, encoder_hidden_dim=256):
        super().__init__()
        self.encoder = CameraEncoder(hidden_dim=encoder_hidden_dim)
        
        # Combined feature size from 3 cameras: 3 * encoder_hidden_dim
        self.fused_dim = 3 * encoder_hidden_dim
        
        # Decoder network: map fused features back to spatial representation and upscale
        self.fc = nn.Sequential(
            nn.Linear(self.fused_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 64 * 8 * 8),  # project to 64 channels, 8x8 spatial grid
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 32 x 16 x 16
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # -> 16 x 32 x 32
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 2, kernel_size=4, stride=2, padding=1),   # -> 2 x 64 x 64
            nn.Sigmoid()  # output probability for each grid cell
        )

    def forward(self, cameras):
        # cameras shape: (batch_size, 3, 3, 64, 64)
        batch_size = cameras.size(0)
        
        # Reshape to process all cameras at once through the encoder
        # Shape: (batch_size * 3, 3, 64, 64)
        flat_cameras = cameras.view(batch_size * 3, 3, 64, 64)
        
        # Extract features: (batch_size * 3, encoder_hidden_dim)
        features = self.encoder(flat_cameras)
        
        # Reshape back to separate camera dimensions
        # Shape: (batch_size, 3, encoder_hidden_dim)
        features = features.view(batch_size, 3, -1)
        
        # Flatten multi-camera features to fuse them: (batch_size, 3 * encoder_hidden_dim)
        fused = features.view(batch_size, -1)
        
        # Decode to spatial representation: (batch_size, 64, 8, 8)
        spatial_features = self.fc(fused).view(batch_size, 64, 8, 8)
        
        # Decode and upscale to BEV grid: (batch_size, 2, 64, 64)
        bev_grid = self.decoder(spatial_features)
        
        return bev_grid
