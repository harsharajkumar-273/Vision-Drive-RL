import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os

class PPOBuffer:
    """
    Buffer to store trajectories for PPO updates.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.states = []       # For oracle: state vectors; for image-based: dicts/lists of observations
        self.bev_grids = []    # Only for BEV agent
        self.camera_obs = []   # Only for End-to-End agent
        self.telemetry = []    # Speed, current steer
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def store(self, action, log_prob, reward, value, done, state=None, bev=None, cameras=None, telemetry=None):
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
        if state is not None:
            self.states.append(state)
        if bev is not None:
            self.bev_grids.append(bev)
        if cameras is not None:
            self.camera_obs.append(cameras)
        if telemetry is not None:
            self.telemetry.append(telemetry)

    def get_tensors(self, device):
        actions = torch.tensor(np.array(self.actions), dtype=torch.float32, device=device)
        log_probs = torch.tensor(np.array(self.log_probs), dtype=torch.float32, device=device)
        rewards = np.array(self.rewards, dtype=np.float32)
        values = torch.tensor(np.array(self.values), dtype=torch.float32, device=device)
        dones = np.array(self.dones, dtype=np.float32)
        
        states = torch.tensor(np.array(self.states), dtype=torch.float32, device=device) if len(self.states) > 0 else None
        bev_grids = torch.tensor(np.array(self.bev_grids), dtype=torch.float32, device=device) if len(self.bev_grids) > 0 else None
        camera_obs = torch.tensor(np.array(self.camera_obs), dtype=torch.float32, device=device) if len(self.camera_obs) > 0 else None
        telemetry = torch.tensor(np.array(self.telemetry), dtype=torch.float32, device=device) if len(self.telemetry) > 0 else None
        
        return actions, log_probs, rewards, values, dones, states, bev_grids, camera_obs, telemetry


class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent.
    Supports Oracle, BEV, and End-to-End network architectures.
    """
    def __init__(
        self, 
        model, 
        agent_type="bev", 
        lr=3e-4, 
        gamma=0.99, 
        gae_lambda=0.95, 
        clip_eps=0.2, 
        c1_val=0.5, 
        c2_ent=0.01, 
        device="cpu"
    ):
        self.model = model.to(device)
        self.agent_type = agent_type  # "oracle", "bev", "end_to_end"
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.c1_val = c1_val
        self.c2_ent = c2_ent
        self.device = device
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.buffer = PPOBuffer()

    def select_action(self, obs, evaluation=False):
        """
        Select action given observations. Returns action (numpy array), log_probability, and critic value.
        """
        self.model.eval()
        with torch.no_grad():
            if self.agent_type == "oracle":
                state = torch.tensor(obs["state"], dtype=torch.float32, device=self.device).unsqueeze(0)
                dist, value = self.model(state)
            elif self.agent_type == "bev":
                # Convert raw observations to PyTorch tensors
                bev = torch.tensor(obs["bev"], dtype=torch.float32, device=self.device).unsqueeze(0)
                telemetry = torch.tensor(obs["telemetry"], dtype=torch.float32, device=self.device).unsqueeze(0)
                dist, value = self.model(bev, telemetry)
            elif self.agent_type == "end_to_end":
                # Transpose cameras from (3, 64, 64, 3) to (3, 3, 64, 64) and convert to float
                cameras = np.transpose(obs["cameras"], (0, 3, 1, 2)).astype(np.float32) / 255.0
                cameras = torch.tensor(cameras, dtype=torch.float32, device=self.device).unsqueeze(0)
                telemetry = torch.tensor(obs["telemetry"], dtype=torch.float32, device=self.device).unsqueeze(0)
                dist, value = self.model(cameras, telemetry)
            
            if evaluation:
                action = dist.mean
            else:
                action = dist.sample()
                
            # Clamp action to gym action space limits [-1.0, 1.0]
            action = torch.clamp(action, -1.0, 1.0)
            log_prob = dist.log_prob(action).sum(dim=-1)
            
        return action.cpu().numpy()[0], log_prob.cpu().item(), value.cpu().item()

    def update(self):
        """
        Performs a PPO policy and value network update.
        """
        self.model.train()
        
        # Get tensors from buffer
        actions, log_probs_old, rewards, values, dones, states, bev_grids, camera_obs, telemetry = self.buffer.get_tensors(self.device)
        
        # Calculate Returns and Advantages using Generalized Advantage Estimation (GAE)
        returns = []
        advantages = []
        gae = 0.0
        
        # Next value starts at 0 if terminal or the last value prediction if truncated
        next_value = 0.0
        
        # Backward pass for GAE
        for step in reversed(range(len(rewards))):
            if step == len(rewards) - 1:
                # Last step
                next_non_terminal = 1.0 - dones[step]
                # delta = r_t + gamma * V(s_{t+1}) - V(s_t)
                delta = rewards[step] - values[step] # simple TD-error baseline if last step
            else:
                next_non_terminal = 1.0 - dones[step]
                delta = rewards[step] + self.gamma * values[step + 1] * next_non_terminal - values[step]
                
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[step].cpu().item())
            
        advantages = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Perform updates (typically 4-10 epochs over the collected rollouts)
        epochs = 5
        batch_size = 64
        dataset_size = len(rewards)
        
        epoch_losses = []
        epoch_val_losses = []
        
        for _ in range(epochs):
            permutation = torch.randperm(dataset_size)
            for i in range(0, dataset_size, batch_size):
                indices = permutation[i:i+batch_size]
                if len(indices) < 8:  # skip too small batches
                    continue
                
                # Fetch batch slices
                b_actions = actions[indices]
                b_log_probs_old = log_probs_old[indices]
                b_advantages = advantages[indices]
                b_returns = returns[indices]
                
                # Forward pass
                if self.agent_type == "oracle":
                    b_states = states[indices]
                    dist, values_pred = self.model(b_states)
                elif self.agent_type == "bev":
                    b_bev = bev_grids[indices]
                    b_telemetry = telemetry[indices]
                    dist, values_pred = self.model(b_bev, b_telemetry)
                elif self.agent_type == "end_to_end":
                    b_cameras = camera_obs[indices]
                    b_telemetry = telemetry[indices]
                    dist, values_pred = self.model(b_cameras, b_telemetry)
                
                values_pred = values_pred.squeeze(-1)
                
                # Calculate log probs and entropy under new policy
                log_probs_new = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()
                
                # Policy ratio: r_t(theta) = pi_new(a|s) / pi_old(a|s)
                ratios = torch.exp(log_probs_new - b_log_probs_old)
                
                # Surrogate losses
                surr1 = ratios * b_advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss: mean squared error
                value_loss = F.mse_loss(values_pred, b_returns)
                
                # Total loss = policy_loss + c1 * value_loss - c2 * entropy
                total_loss = policy_loss + self.c1_val * value_loss - self.c2_ent * entropy
                
                # Optimization step
                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                epoch_losses.append(policy_loss.item())
                epoch_val_losses.append(value_loss.item())
                
        # Clear buffer
        self.buffer.reset()
        
        return np.mean(epoch_losses), np.mean(epoch_val_losses)

    def save(self, filepath):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "agent_type": self.agent_type
        }, filepath)

    def load(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.agent_type = checkpoint["agent_type"]
