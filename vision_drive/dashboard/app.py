import os
import sys
import time
import math
import numpy as np
import torch
from flask import Flask, render_template, jsonify
from flask_socketio import SocketIO, emit
from PIL import Image
import io
import base64

# Add workspace to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import gymnasium as gym
import vision_drive
from vision_drive.env.driving_env import BEVDrivingEnv
from vision_drive.agent.models import BEVActorCritic
from vision_drive.agent.ppo import PPOAgent
from vision_drive.utils.logger import TrajectoryLogger

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Global variables to manage simulation state
env = None
agent = None
manual_mode = False
active_episode = False
control_action = [0.0, 0.0]  # [steer, throttle]
current_obs = None
last_step_time = 0

# Data logging variables
logger = TrajectoryLogger()
is_recording = False

# Path to trained model
MODEL_PATH = "results/bev_agent_latest.pt"

def array_to_base64(arr):
    """Converts a numpy RGB array to a base64 encoded PNG string."""
    img = Image.fromarray(arr.astype(np.uint8))
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_str}"

def process_bev_to_rgb(bev_grid):
    """Converts a 2-channel BEV occupancy grid to an RGB numpy array for display."""
    # bev_grid shape: (2, 64, 64)
    # Channel 0: Road (green)
    # Channel 1: Obstacles (red)
    h, w = bev_grid.shape[1], bev_grid.shape[2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    
    # Road channel: set green intensity
    rgb[..., 1] = (bev_grid[0] * 120).astype(np.uint8)  # soft green
    
    # Obstacle channel: set red intensity
    rgb[..., 0] = (bev_grid[1] * 230).astype(np.uint8)  # soft red
    
    # Add ego vehicle representation in the bottom center
    # Ego vehicle is located at (u=32, v=58)
    ego_u = w // 2
    ego_v = int(h * 30.0 / 32.0) # ~58
    
    # Draw simple bounding box for ego car (1.8m width, 4.5m length)
    # at 0.5m/px resolution: width is ~4px, length is ~9px
    l_half, w_half = 4, 2
    rgb[ego_v - l_half:ego_v + l_half + 1, ego_u - w_half:ego_u + w_half + 1, 2] = 255  # blue
    
    # Upscale for better viewing in web app (e.g. 256x256)
    img = Image.fromarray(rgb)
    img_large = img.resize((256, 256), Image.Resampling.NEAREST)
    return np.array(img_large)

def process_cameras_to_base64(cameras):
    # cameras shape: (3, 64, 64, 3)
    # Resize camera frames to look better
    left_img = Image.fromarray(cameras[1])
    center_img = Image.fromarray(cameras[0])
    right_img = Image.fromarray(cameras[2])
    
    new_size = (256, 256)
    left_img = left_img.resize(new_size, Image.Resampling.BILINEAR)
    center_img = center_img.resize(new_size, Image.Resampling.BILINEAR)
    right_img = right_img.resize(new_size, Image.Resampling.BILINEAR)
    
    return {
        "left": array_to_base64(np.array(left_img)),
        "center": array_to_base64(np.array(center_img)),
        "right": array_to_base64(np.array(right_img))
    }

def init_simulation():
    global env, agent, current_obs
    env = BEVDrivingEnv()
    
    # Try to load pre-trained agent if exists, otherwise initialize randomly
    model = BEVActorCritic()
    agent = PPOAgent(model, agent_type="bev")
    
    if os.path.exists(MODEL_PATH):
        try:
            agent.load(MODEL_PATH)
            print("Loaded trained model from", MODEL_PATH)
        except Exception as e:
            print("Could not load trained model, running random policy:", e)
            
    current_obs, _ = env.reset()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    print('Client connected')
    emit('status', {'data': 'Connected to Simulator Backend'})

@socketio.on('start_simulation')
def start_simulation():
    global active_episode, current_obs
    print("Starting simulation episode...")
    init_simulation()
    active_episode = True
    send_sim_state()

@socketio.on('stop_simulation')
def stop_simulation():
    global active_episode
    print("Stopping simulation...")
    active_episode = False

@socketio.on('set_mode')
def set_mode(data):
    global manual_mode
    mode = data.get("mode", "ai")
    manual_mode = (mode == "manual")
    print(f"Driver mode set to: {'MANUAL' if manual_mode else 'AI'}")

@socketio.on('key_action')
def key_action(data):
    global control_action
    # Map WASD/Arrows to steering and acceleration
    steer = data.get("steer", 0.0)      # -1.0 to 1.0
    throttle = data.get("throttle", 0.0) # -1.0 to 1.0
    control_action = [steer, throttle]

@socketio.on('start_recording')
def start_recording():
    global is_recording
    is_recording = True
    logger.start()
    print("Recording started in backend.")
    emit('recording_status', {'status': 'recording'})

@socketio.on('stop_recording')
def stop_recording():
    global is_recording
    if is_recording:
        filepath = logger.stop_and_save()
        is_recording = False
        print(f"Recording stopped in backend. Saved to {filepath}")
        emit('recording_status', {'status': 'idle', 'filepath': filepath})
    else:
        emit('recording_status', {'status': 'idle'})

def send_sim_state():
    global env, current_obs, last_step_time, control_action, active_episode
    if env is None or not active_episode:
        return
        
    t_start = time.time()
    
    # Compute action
    if manual_mode:
        action = np.array(control_action, dtype=np.float32)
    else:
        # Pass observations to PPO agent
        bev_grid = env.get_bev_grid()
        agent_obs = {
            "bev": bev_grid,
            "telemetry": current_obs["telemetry"]
        }
        action, _, _ = agent.select_action(agent_obs, evaluation=True)
        
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Record step if recording is active
    if is_recording:
        logger.record_step(
            cameras=current_obs["cameras"],  # record the camera observation before taking action
            telemetry=current_obs["telemetry"],
            action=action,
            reward=reward,
            done=terminated or truncated,
            bev_grid=env.get_bev_grid()
        )
        
    current_obs = obs
    
    # Format and send states
    cameras_b64 = process_cameras_to_base64(obs["cameras"])
    bev_grid = env.get_bev_grid()
    bev_b64 = array_to_base64(process_bev_to_rgb(bev_grid))
    
    # Send telemetry & metrics
    state_payload = {
        "cameras": cameras_b64,
        "bev": bev_b64,
        "speed": float(info["speed"]),
        "steering": float(info["steering"]),
        "lane_offset": float(info["lane_offset"]),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "action": [float(action[0]), float(action[1])],
        "manual_mode": manual_mode
    }
    
    socketio.emit('sim_update', state_payload)
    
    if terminated or truncated:
        print("Episode ended. Resetting environment...")
        current_obs, _ = env.reset()
        socketio.emit('episode_ended', {'reason': 'collision' if info.get('collided') else 'out_of_road' if info.get('out_of_road') else 'timeout'})

    # Schedule next step if still active
    if active_episode:
        # Target 10 Hz simulation loop (0.1 seconds dt)
        elapsed = time.time() - t_start
        delay = max(0.01, 0.1 - elapsed)
        socketio.sleep(delay)
        socketio.start_background_task(send_sim_state)

if __name__ == '__main__':
    print("Initializing Simulation Web Server...")
    init_simulation()
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
