import os
import argparse
import gymnasium as gym
import numpy as np
import torch
import time

import vision_drive
from vision_drive.env.driving_env import BEVDrivingEnv
from vision_drive.agent.models import BEVActorCritic
from vision_drive.agent.ppo import PPOAgent

def parse_args():
    parser = argparse.ArgumentParser(description="Train PPO Agent on BEVDriving-v0")
    parser.add_argument("--steps", type=int, default=30000, help="Total training steps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--save-path", type=str, default="results/bev_agent_latest.pt", help="Path to save trained model")
    parser.add_argument("--curved", action="store_true", default=True, help="Train on curved roads")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu or cuda)")
    return parser.parse_args()

def train():
    args = parse_args()
    
    # Device setup
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # Initialize environment
    config = {"curved": args.curved}
    env = BEVDrivingEnv(config=config)
    
    # Initialize model and agent
    model = BEVActorCritic()
    agent = PPOAgent(
        model=model,
        agent_type="bev",
        lr=args.lr,
        device=device
    )
    
    # Training state variables
    total_steps = 0
    episode_count = 0
    update_every_steps = 1024
    
    # Telemetry logging
    episode_rewards = []
    episode_lengths = []
    collisions = []
    
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    print("\nStarting PPO training loop...")
    print("=" * 60)
    
    start_time = time.time()
    
    while total_steps < args.steps:
        obs, info = env.reset()
        ep_reward = 0
        ep_steps = 0
        done = False
        
        while not done:
            # Prepare observation for BEV agent
            bev_grid = env.get_bev_grid()
            agent_obs = {
                "bev": bev_grid,
                "telemetry": obs["telemetry"]
            }
            
            # Select action
            action, log_prob, value = agent.select_action(agent_obs, evaluation=False)
            
            # Step environment
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store in agent buffer
            agent.buffer.store(
                action=action,
                log_prob=log_prob,
                reward=reward,
                value=value,
                done=terminated,  # don't boostrap on termination
                bev=bev_grid,
                telemetry=obs["telemetry"]
            )
            
            obs = next_obs
            ep_reward += reward
            ep_steps += 1
            total_steps += 1
            
            # Update policy periodically
            if total_steps % update_every_steps == 0:
                policy_loss, val_loss = agent.update()
                elapsed_time = time.time() - start_time
                print(f"Step: {total_steps}/{args.steps} | "
                      f"Policy Loss: {policy_loss:.4f} | "
                      f"Value Loss: {val_loss:.4f} | "
                      f"FPS: {total_steps / elapsed_time:.1f}")
                
        episode_count += 1
        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_steps)
        collisions.append(float(info.get("collided", False)))
        
        # Print episode summary periodically
        if episode_count % 10 == 0:
            avg_rew = np.mean(episode_rewards[-10:])
            avg_len = np.mean(episode_lengths[-10:])
            col_rate = np.mean(collisions[-10:]) * 100
            print(f"Episode: {episode_count} | Avg Reward: {avg_rew:.2f} | "
                  f"Avg Length: {avg_len:.1f} | Collision Rate: {col_rate:.1f}%")
            
    print("=" * 60)
    print(f"Training completed in {time.time() - start_time:.1f} seconds.")
    print(f"Saving latest checkpoint to: {args.save_path}")
    agent.save(args.save_path)
    env.close()

if __name__ == "__main__":
    train()
