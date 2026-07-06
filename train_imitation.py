import os
import glob
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from vision_drive.env.driving_env import BEVDrivingEnv
from vision_drive.agent.models import EndToEndActorCritic
from run_benchmark import HeuristicOracleDriver
from vision_drive.utils.logger import TrajectoryLogger

class DemonstrationDataset(Dataset):
    """
    PyTorch Dataset for behavioral cloning.
    Loads camera frames, telemetry, and actions from demonstration pickles.
    """
    def __init__(self, filepaths):
        self.cameras = []
        self.telemetry = []
        self.actions = []

        for filepath in filepaths:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                
            # Transpose cameras from (N, 3, 64, 64, 3) to (N, 3, 3, 64, 64)
            # and normalize values to [0, 1]
            cams = data["cameras"] # Shape: (N, 3, 64, 64, 3)
            # transpose: (N, 3, 3, 64, 64) where transposed axis is index 4 (channels) to index 2
            cams_transposed = np.transpose(cams, (0, 1, 4, 2, 3)).astype(np.float32) / 255.0
            
            self.cameras.append(cams_transposed)
            self.telemetry.append(data["telemetry"])
            self.actions.append(data["actions"])

        self.cameras = np.concatenate(self.cameras, axis=0)
        self.telemetry = np.concatenate(self.telemetry, axis=0)
        self.actions = np.concatenate(self.actions, axis=0)
        
        print(f"Loaded dataset with {self.cameras.shape[0]} total state-action transitions.")

    def __len__(self):
        return self.cameras.shape[0]

    def __getitem__(self, idx):
        return {
            "cameras": torch.tensor(self.cameras[idx], dtype=torch.float32),
            "telemetry": torch.tensor(self.telemetry[idx], dtype=torch.float32),
            "action": torch.tensor(self.actions[idx], dtype=torch.float32)
        }

def generate_synthetic_demos(num_episodes=5, save_dir="results/demonstrations"):
    """
    Generates demonstration files using the heuristic oracle controller.
    Useful for testing the training script before human teleoperation data is collected.
    """
    print(f"Generating {num_episodes} episodes of synthetic demonstration data...")
    env = BEVDrivingEnv(config={"curved": True})
    oracle = HeuristicOracleDriver()
    logger = TrajectoryLogger(save_dir=save_dir)
    
    for ep in range(num_episodes):
        obs, info = env.reset(seed=100 + ep)
        logger.start()
        done = False
        
        while not done:
            action = oracle.select_action(env)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            logger.record_step(
                cameras=obs["cameras"],
                telemetry=obs["telemetry"],
                action=action,
                reward=reward,
                done=done,
                bev_grid=env.get_bev_grid()
            )
            obs = next_obs
            
        logger.stop_and_save()
    
    env.close()
    print("Synthetic demonstration generation complete.\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Train Behavioral Cloning Policy (Imitation Learning)")
    parser.add_argument("--demo-dir", type=str, default="results/demonstrations", help="Directory containing demonstrations")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="results/imitation_agent.pt", help="Path to save trained policy")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on (cpu or cuda)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training imitation policy on: {device}")

    # Check for demonstration files
    demo_files = glob.glob(os.path.join(args.demo_dir, "*.pkl"))
    if len(demo_files) == 0:
        print("Warning: No demonstration files found in", args.demo_dir)
        generate_synthetic_demos(num_episodes=5, save_dir=args.demo_dir)
        demo_files = glob.glob(os.path.join(args.demo_dir, "*.pkl"))

    # Load dataset
    dataset = DemonstrationDataset(demo_files)
    
    # Train/Val split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = EndToEndActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    print("\nStarting Behavioral Cloning training...")
    print("=" * 50)
    
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            cameras = batch["cameras"].to(device)
            telemetry = batch["telemetry"].to(device)
            target_actions = batch["action"].to(device)
            
            # Predict actions using policy (actor distribution mean)
            dist, _ = model(cameras, telemetry)
            pred_actions = dist.mean
            
            loss = criterion(pred_actions, target_actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * cameras.size(0)
            
        train_loss /= len(train_dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                cameras = batch["cameras"].to(device)
                telemetry = batch["telemetry"].to(device)
                target_actions = batch["action"].to(device)
                
                dist, _ = model(cameras, telemetry)
                pred_actions = dist.mean
                
                loss = criterion(pred_actions, target_actions)
                val_loss += loss.item() * cameras.size(0)
                
        val_loss /= len(val_dataset)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train MSE Loss: {train_loss:.5f} | Val MSE Loss: {val_loss:.5f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Format checkpoint matching PPOAgent saving style
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "agent_type": "end_to_end"  # Imitation uses End-to-End camera inputs
            }, args.save_path)
            
    print("=" * 50)
    print(f"Imitation training complete. Best Val MSE: {best_val_loss:.5f}")
    print(f"Model saved to: {args.save_path}")

if __name__ == "__main__":
    main()
