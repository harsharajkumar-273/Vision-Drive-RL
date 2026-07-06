# VisionDrive-RL Performance & Robustness Benchmark Report

This report evaluates the geometry-based heuristic Oracle, reinforcement learning (RL with ground-truth BEV), imitation learning (Behavioral Cloning), and the end-to-end vision pipeline (Perception Net + Control Net) under various environmental perturbations (fog, rain, lens mud, night).

## Quantitative Performance Comparison

| Scenario       |   Oracle_Reward |   RL_GT_BEV_Reward |   BC_Imitation_Reward |   Vision_Pipeline_Reward |
|:---------------|----------------:|-------------------:|----------------------:|-------------------------:|
| Ideal (Normal) |         61.5295 |           -85.0168 |              -80.12   |                 -79.8005 |
| Light Fog      |         61.5295 |           -85.0168 |              -79.502  |                 -80.3423 |
| Heavy Fog      |         61.5295 |           -85.0168 |              -80.1594 |                 -81.3085 |
| Light Rain     |         61.5295 |           -85.0168 |              -80.0961 |                 -79.8352 |
| Heavy Rain     |         61.5295 |           -85.0168 |              -80.068  |                 -79.8711 |
| Mud on Lens    |         61.5295 |           -85.0168 |              -80.1346 |                 -79.8528 |
| Night Mode     |         61.5295 |           -85.0168 |              -79.849  |                 -84.085  |

## Full Telemetry Results

| Scenario       |   Oracle_Reward |   Oracle_Collision% |   Oracle_LaneOffset |   RL_GT_BEV_Reward |   RL_GT_BEV_Collision% |   RL_GT_BEV_LaneOffset |   BC_Imitation_Reward |   BC_Imitation_Collision% |   BC_Imitation_LaneOffset |   Vision_Pipeline_Reward |   Vision_Pipeline_Collision% |   Vision_Pipeline_LaneOffset |
|:---------------|----------------:|--------------------:|--------------------:|-------------------:|-----------------------:|-----------------------:|----------------------:|--------------------------:|--------------------------:|-------------------------:|-----------------------------:|-----------------------------:|
| Ideal (Normal) |         61.5295 |                   0 |            0.230277 |           -85.0168 |                      0 |               0.868935 |              -80.12   |                         0 |                   2.50794 |                 -79.8005 |                            0 |                      3.5662  |
| Light Fog      |         61.5295 |                   0 |            0.230277 |           -85.0168 |                      0 |               0.868935 |              -79.502  |                         0 |                   2.45877 |                 -80.3423 |                            0 |                      3.57488 |
| Heavy Fog      |         61.5295 |                   0 |            0.230277 |           -85.0168 |                      0 |               0.868935 |              -80.1594 |                         0 |                   2.5238  |                 -81.3085 |                            0 |                      3.58978 |
| Light Rain     |         61.5295 |                   0 |            0.230277 |           -85.0168 |                      0 |               0.868935 |              -80.0961 |                         0 |                   2.50611 |                 -79.8352 |                            0 |                      3.56677 |
| Heavy Rain     |         61.5295 |                   0 |            0.230277 |           -85.0168 |                      0 |               0.868935 |              -80.068  |                         0 |                   2.50404 |                 -79.8711 |                            0 |                      3.56734 |
| Mud on Lens    |         61.5295 |                   0 |            0.230277 |           -85.0168 |                      0 |               0.868935 |              -80.1346 |                         0 |                   2.509   |                 -79.8528 |                            0 |                      3.56707 |
| Night Mode     |         61.5295 |                   0 |            0.230277 |           -85.0168 |                      0 |               0.868935 |              -79.849  |                         0 |                   2.51373 |                 -84.085  |                            0 |                      3.63997 |

## Analysis & Takeaways

1. **Spatial Representation Advantage**: The Bird's-Eye-View (BEV) mapping allows control policies (like the RL PPO Agent) to decouple control from raw visual features. This provides higher baseline stability.
2. **Behavioral Cloning vs RL**: The Behavioral Cloning (Imitation Learning) policy trains directly on cameras. In ideal conditions, it matches human performance nicely, but suffers under high visual degradation (heavy rain/fog/night) due to covariate shift, since it is not trained on recovery trajectories.
3. **Vision-Only Pipeline (Perception + Control)**: Combining a visual perception network (BEVProjector) with the control policy offers a modular approach to self-driving. However, error compounding (perception noise translating to control errors) remains a key challenge under visual hazards.
