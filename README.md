# VisionDrive-RL: Vision-Only Autonomous Driving via Bird's-Eye-View (BEV) Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![Gymnasium](https://img.shields.io/badge/Gymnasium-1.0-008080.svg)](https://gymnasium.farama.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

VisionDrive-RL is a modular, high-performance reinforcement learning (RL) framework designed to simulate Tesla's vision-only autonomous driving paradigm. It takes inputs from multiple perspective camera feeds, fuses and projects them into a unified 2D Bird's-Eye-View (BEV) occupancy grid, and trains an actor-critic policy (PPO) to control steering, throttle, and braking.

This repository serves as a portfolio-grade demonstration of deep reinforcement learning, computer vision perception, spatial coordinate transforms, and real-time visualization systems.

---

## 🚘 System Architecture

```mermaid
graph TD
    %% Camera Input Section
    subgraph Multi-View Perspective Cameras
        CamL["Left Camera (-30° Yaw)"]
        CamC["Center Camera (0° Yaw)"]
        CamR["Right Camera (+30° Yaw)"]
    end

    %% Perception Pipeline
    subgraph Visual Perception Network
        Encoder["CNN Feature Extractor<br>(ResNet Backbone)"]
        BEVProj["BEV Projection Decoder<br>(ConvTranspose2d)"]
        OccupancyGrid["BEV Occupancy Map<br>(2x64x64 Grid)"]
    end

    %% Control Pipeline
    subgraph Reinforcement Learning Planner
        PPOActor["PPO Actor Network<br>(Steering & Throttle)"]
        PPOCritic["PPO Critic Network<br>(Value Prediction)"]
        Telemetry["Telemetry Feed<br>(Speed, Steer Angle)"]
    end

    %% Output Loop
    subgraph Env["BEVDriving-v0 Environment"]
        Physics["Ego Vehicle Physics<br>(Bicycle Model)"]
        Traffic["Dynamic Traffic Cars<br>& Obstacles"]
    end

    %% Mapping Connections
    CamL --> Encoder
    CamC --> Encoder
    CamR --> Encoder
    Encoder --> BEVProj
    BEVProj --> OccupancyGrid
    
    OccupancyGrid --> PPOActor
    Telemetry --> PPOActor
    OccupancyGrid --> PPOCritic
    Telemetry --> PPOCritic
    
    PPOActor -->|Action: [Steer, Accel]| Physics
    Physics -->|Updates| Traffic
    Traffic -->|Generates Next State| Multi-View Perspective Cameras
    Physics -->|Updates Telemetry| Telemetry
```

---

## 🌟 Key Features

1. **Deterministic Perspective Rendering Engine**: Standard perspective projection equations ($u = f \cdot x/z$, $v = f \cdot y/z$) render Left, Center, and Right camera inputs headlessly in pure Python/NumPy, bypassing heavy game engine dependencies.
2. **BEV Occupancy Network**: Maps multiple $64\times 64$ perspective cameras to a unified $64\times 64$ orthographic grid mapping road bounds (Channel 0) and dynamic obstacles (Channel 1).
3. **PPO RL Policy**: A custom PyTorch Proximal Policy Optimization implementation driving a continuous-action space vehicle.
4. **Interactive Dashboard**: A Flask-SocketIO web server streaming real-time simulation frames (perspective views, BEV grid, telemetry) with manual override controls.
5. **Robustness Benchmark Suite**: Compares control performance under simulated environmental hazards like heavy rain, fog, lens dirt/occlusion, and night conditions.

---

## 🛠️ Quick Start

### 1. Installation
Ensure you are using Python 3.10+ and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Launch the Web Dashboard
Experience the simulation visually! Run the dashboard server:
```bash
python vision_drive/dashboard/app.py
```
Open [http://localhost:5000](http://localhost:5000) in your web browser.
* **AI Control**: Let the PPO policy drive the car along lanes.
* **Manual Override**: Press **Manual Override**, and steer using `W`, `A`, `S`, `D` or Arrow keys.

### 3. Train the PPO Policy
Train the BEV RL agent from scratch:
```bash
python train.py --steps 30000 --device cpu
```
This trains the actor-critic model and saves the checkpoint to `results/bev_agent_latest.pt`.

### 4. Run the Robustness Benchmarks
Evaluate the trained agent against an Oracle controller under simulated visual hazards:
```bash
python run_benchmark.py --model-path results/bev_agent_latest.pt --episodes 15
```
This saves:
* A quantitative performance table in `results/benchmark_report.md`.
* A visualization comparing performance in `results/robustness_comparison.png`.

---

## 📊 Evaluation & Robustness Analysis

We compare two control strategies across multiple scenarios:
* **Heuristic Oracle**: An ideal lane-follower driver with direct access to environment coordinates.
* **BEV PPO Agent**: The RL agent trained on the spatial BEV occupancy grid.

### Robustness Test Scenarios
* **Fog/Rain**: Adds visual Gaussian blur and noise to camera streams, testing the perception model's feature extraction stability.
* **Mud on Lens**: Simulates localized visual occlusions on individual cameras to evaluate policy redundancy.
* **Night Driving**: Darkens perspective frames, keeping only the headlight cone lit, verifying low-light perception.

---

## 📂 Project Structure

```
vision_drive/
├── env/
│   └── driving_env.py        # Gym environment (cameras, dynamics, rewards)
├── perception/
│   ├── encoder.py            # ResNet-like CNN camera encoder
│   └── bev_projector.py      # Decoder creating BEV grid from 3 cameras
├── agent/
│   ├── models.py             # Actor-Critic network models (Oracle, BEV, E2E)
│   └── ppo.py                # PyTorch PPO RL algorithm
├── dashboard/
│   ├── app.py                # Flask-SocketIO live streaming webapp
│   └── templates/
│       └── index.html        # Glassmorphic UI control console
└── utils/
    └── metrics.py            # Loggers & telemetry trackers
```

---

## 📜 License
This project is licensed under the MIT License.
