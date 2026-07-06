import os
import glob
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from vision_drive.env.driving_env import BEVDrivingEnv
from vision_drive.perception.bev_projector import BEVProjector
from train_imitation import generate_synthetic_demos

class PerceptionDataset(Dataset):
    """
    PyTorch Dataset for training the BEV projection network.
    Loads camera frames and target ground-truth BEV grids.
    """
    def __init__(self, filepaths):
        self.cameras = []
        self.bev_grids = []

        for filepath in filepaths:
            with open(filepath, "rb") as f:
                data = pickle.load(f)
                
            if "bev_grids" not in data:
                print(f"Skipping {filepath} as it does not contain BEV occupancy grids.")
                continue

            # Transpose cameras from (N, 3, 64, 64, 3) to (N, 3, 3, 64, 64)
            # and normalize to [0, 1]
            cams = data["cameras"] # Shape: (N, 3, 64, 64, 3)
            cams_transposed = np.transpose(cams, (0, 1, 4, 2, 3)).astype(np.float32) / 255.0
            
            self.cameras.append(cams_transposed)
            self.bev_grids.append(data["bev_grids"])

        if len(self.cameras) == 0:
            raise ValueError("No valid demonstration data with BEV grids found.")

        self.cameras = np.concatenate(self.cameras, axis=0)
        self.bev_grids = np.concatenate(self.bev_grids, axis=0)
        
        print(f"Loaded dataset with {self.cameras.shape[0]} total camera-to-BEV frames.")

    def __len__(self):
        return self.cameras.shape[0]

    def __getitem__(self, idx):
        return {
            "cameras": torch.tensor(self.cameras[idx], dtype=torch.float32),
            "bev_grid": torch.tensor(self.bev_grids[idx], dtype=torch.float32)
        }

def parse_args():
    parser = argparse.ArgumentParser(description="Train Visual Perception BEV Projector Network")
    parser.add_argument("--demo-dir", type=str, default="results/demonstrations", help="Directory containing demonstrations")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="results/bev_projector.pt", help="Path to save trained projection network")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on (cpu or cuda)")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training perception projector on: {device}")

    # Check for demonstration files
    demo_files = glob.glob(os.path.join(args.demo_dir, "*.pkl"))
    if len(demo_files) == 0:
        print("Warning: No demonstration files found in", args.demo_dir)
        generate_synthetic_demos(num_episodes=5, save_dir=args.demo_dir)
        demo_files = glob.glob(os.path.join(args.demo_dir, "*.pkl"))

    # Load dataset
    try:
        dataset = PerceptionDataset(demo_files)
    except ValueError as e:
        print(f"Error loading dataset: {e}. Re-generating synthetic demos...")
        generate_synthetic_demos(num_episodes=5, save_dir=args.demo_dir)
        demo_files = glob.glob(os.path.join(args.demo_dir, "*.pkl"))
        dataset = PerceptionDataset(demo_files)
    
    # Train/Val split (80/20)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Initialize model
    model = BEVProjector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()  # Binary Cross-Entropy Loss for grid probability estimation

    print("\nStarting BEV Projection network training...")
    print("=" * 50)
    
    best_val_loss = float("inf")
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        for batch in train_loader:
            cameras = batch["cameras"].to(device)
            target_grids = batch["bev_grid"].to(device)
            
            # Predict BEV grid
            pred_grids = model(cameras)
            
            loss = criterion(pred_grids, target_grids)
            
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
                target_grids = batch["bev_grid"].to(device)
                
                pred_grids = model(cameras)
                loss = criterion(pred_grids, target_grids)
                
                val_loss += loss.item() * cameras.size(0)
                
        val_loss /= len(val_dataset)
        
        print(f"Epoch {epoch+1:02d}/{args.epochs:02d} | Train BCE Loss: {train_loss:.5f} | Val BCE Loss: {val_loss:.5f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict()
            }, args.save_path)
            
    print("=" * 50)
    print(f"Perception training complete. Best Val BCE Loss: {best_val_loss:.5f}")
    print(f"Model saved to: {args.save_path}")

if __name__ == "__main__":
    main()
