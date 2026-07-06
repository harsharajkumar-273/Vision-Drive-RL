from gymnasium.envs.registration import register

register(
    id="BEVDriving-v0",
    entry_point="vision_drive.env.driving_env:BEVDrivingEnv",
    max_episode_steps=200,
)
