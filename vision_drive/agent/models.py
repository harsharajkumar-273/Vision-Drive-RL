import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from vision_drive.perception.encoder import CameraEncoder

class OracleActorCritic(nn.Module):
    """
    RL Policy using ground-truth environment state vectors (no images).
    Acts as an oracle baseline.
    """
    def __init__(self, state_dim=8, action_dim=2):
        super().__init__()
        # Actor network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim * 2)  # outputs mean and log_std for each action
        )
        # Critic network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        # State: (batch_size, state_dim)
        actor_out = self.actor(state)
        mean, log_std = torch.chunk(actor_out, 2, dim=-1)
        mean = torch.tanh(mean)  # Bound action means to [-1, 1] to prevent gradient collapse on clamp
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        
        value = self.critic(state)
        
        dist = Normal(mean, std)
        return dist, value


class BEVActorCritic(nn.Module):
    """
    RL Policy using the Bird's-Eye-View (BEV) representation (2, 64, 64).
    """
    def __init__(self, action_dim=2):
        super().__init__()
        # CNN to process BEV occupancy map (2 channels: road, obstacles)
        self.bev_cnn = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=4, stride=2, padding=1),  # -> 16 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1), # -> 32 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1), # -> 64 x 8 x 8
            nn.ReLU(),
            nn.Flatten()  # -> 64 * 8 * 8 = 4096
        )
        
        # Ego state MLP (speed, current steer)
        self.ego_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        
        # Actor
        self.actor_fc = nn.Sequential(
            nn.Linear(4096 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim * 2)
        )
        
        # Critic
        self.critic_fc = nn.Sequential(
            nn.Linear(4096 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, bev, ego_state):
        # bev shape: (batch_size, 2, 64, 64)
        # ego_state shape: (batch_size, 2)
        bev_features = self.bev_cnn(bev)
        ego_features = self.ego_mlp(ego_state)
        
        # Fuse spatial and telemetry features
        fused = torch.cat([bev_features, ego_features], dim=-1)
        
        actor_out = self.actor_fc(fused)
        mean, log_std = torch.chunk(actor_out, 2, dim=-1)
        mean = torch.tanh(mean)  # Bound action means to [-1, 1] to prevent gradient collapse on clamp
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        
        value = self.critic_fc(fused)
        
        dist = Normal(mean, std)
        return dist, value


class EndToEndActorCritic(nn.Module):
    """
    RL Policy processing 3 perspective camera frames directly.
    """
    def __init__(self, action_dim=2):
        super().__init__()
        # Camera visual feature extractor
        self.encoder = CameraEncoder(hidden_dim=256)
        
        # Ego state MLP (speed, current steer)
        self.ego_mlp = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU()
        )
        
        # Actor
        self.actor_fc = nn.Sequential(
            nn.Linear(256 * 3 + 32, 256),  # 3 cameras
            nn.ReLU(),
            nn.Linear(256, action_dim * 2)
        )
        
        # Critic
        self.critic_fc = nn.Sequential(
            nn.Linear(256 * 3 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, cameras, ego_state):
        # cameras shape: (batch_size, 3, 3, 64, 64)
        # ego_state shape: (batch_size, 2)
        batch_size = cameras.size(0)
        
        # Process each camera frame
        flat_cameras = cameras.view(batch_size * 3, 3, 64, 64)
        cam_features = self.encoder(flat_cameras)
        
        # Concatenate camera features: (batch_size, 3 * 256)
        fused_cam = cam_features.view(batch_size, -1)
        
        ego_features = self.ego_mlp(ego_state)
        
        # Fuse spatial and telemetry features
        fused = torch.cat([fused_cam, ego_features], dim=-1)
        
        actor_out = self.actor_fc(fused)
        mean, log_std = torch.chunk(actor_out, 2, dim=-1)
        mean = torch.tanh(mean)  # Bound action means to [-1, 1] to prevent gradient collapse on clamp
        log_std = torch.clamp(log_std, min=-20, max=2)
        std = torch.exp(log_std)
        
        value = self.critic_fc(fused)
        
        dist = Normal(mean, std)
        return dist, value
