import os
import pickle
import time
import numpy as np

class TrajectoryLogger:
    """
    Utility class to log and save trajectory data (observations, actions, rewards)
    from human manual teleoperation or oracle sessions.
    """
    def __init__(self, save_dir="results/demonstrations"):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self.reset()

    def reset(self):
        self.cameras = []
        self.telemetry = []
        self.bev_grids = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.timestamps = []
        self.is_recording = False
        self.start_time = None

    def start(self):
        self.reset()
        self.is_recording = True
        self.start_time = time.time()
        print("Trajectory logging started.")

    def record_step(self, cameras, telemetry, action, reward, done, bev_grid=None):
        if not self.is_recording:
            return
        
        # Expecting cameras: np.array of shape (3, 64, 64, 3)
        # Expecting telemetry: np.array of shape (2,)
        # Expecting action: np.array of shape (2,)
        self.cameras.append(cameras)
        self.telemetry.append(telemetry)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.timestamps.append(time.time() - self.start_time)
        if bev_grid is not None:
            self.bev_grids.append(bev_grid)

    def stop_and_save(self):
        if not self.is_recording or len(self.actions) == 0:
            self.is_recording = False
            return None

        self.is_recording = False
        duration = time.time() - self.start_time
        
        trajectory_data = {
            "cameras": np.array(self.cameras, dtype=np.uint8),          # Shape: (N, 3, 64, 64, 3)
            "telemetry": np.array(self.telemetry, dtype=np.float32),    # Shape: (N, 2)
            "actions": np.array(self.actions, dtype=np.float32),        # Shape: (N, 2)
            "rewards": np.array(self.rewards, dtype=np.float32),        # Shape: (N,)
            "dones": np.array(self.dones, dtype=np.bool_),              # Shape: (N,)
            "timestamps": np.array(self.timestamps, dtype=np.float32),  # Shape: (N,)
            "metadata": {
                "num_steps": len(self.actions),
                "duration_seconds": duration,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        if len(self.bev_grids) > 0:
            trajectory_data["bev_grids"] = np.array(self.bev_grids, dtype=np.float32)  # Shape: (N, 2, 64, 64)

        import uuid
        filename = f"teleop_demo_{int(time.time())}_{uuid.uuid4().hex[:6]}.pkl"
        filepath = os.path.join(self.save_dir, filename)
        
        with open(filepath, "wb") as f:
            pickle.dump(trajectory_data, f)
            
        print(f"Saved {len(self.actions)} steps of demonstration to: {filepath}")
        return filepath
