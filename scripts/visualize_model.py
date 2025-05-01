import argparse
from control.ObservationStackingWrapper import ObservationStackingWrapper
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from control.GymEnv import GymEnv
import os
import h5py
import numpy as np
import random

def visualize_model(data_file, lat, lon, config_file, model_file, render_mode, save, num_stack: int = 4, use_fixed_position: bool = True):
    """
    Visualize a trained model by running it in a custom environment.

    Args:
        data_file (str): Path to the data file.
        lat (float): Latitude for the environment.
        lon (float): Longitude for the environment.
        config_file (str): Path to the configuration file.
        model_file (str): Path to the trained model file.
        render_mode (str): Rendering mode ('human' or 'ansi').
    """
    # Initialize the environment
    env = GymEnv(data_file, lat, lon, config_file)

    if use_fixed_position:
        env.use_fixed_position(lat, lon)

    if num_stack > 1:
        env = ObservationStackingWrapper(env, num_stack=num_stack)

    # Load the trained model
    model = PPO.load(model_file, env)

    # Run a single test episode
    obs, _ = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        action, _ = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)
        if render_mode == "ansi":
            if env.unwrapped:
                env.unwrapped.render(mode="ansi")
            else:
                env.render(mode="ansi")
    if render_mode == "human":
        show = not save
        if env.unwrapped:
            env.unwrapped.render(mode="human", show=show)
        else:
            env.render(mode="human", show=show)

    env.close()

def randomize_cutter_position(data_file: str) -> tuple[float, float]:
    if not os.path.exists(data_file):
        raise ValueError("Data file does not exist")
    with h5py.File(data_file, 'r') as data:
        latitudes = data["step_1/current/latitude"]
        longitudes = data["step_1/current/longitude"]
        land_mask = data["step_1/depth/mask"][0]

        if len(latitudes) == 0 or len(longitudes) == 0:
            raise ValueError("Latitude or Longitude data is empty.")

        lat_min, lat_max = np.min(latitudes), np.max(latitudes)
        lon_min, lon_max = np.min(longitudes), np.max(longitudes)

        def is_valid_position(lat, lon):
            lat_idx = (np.abs(latitudes - lat)).argmin()
            lon_idx = (np.abs(longitudes - lon)).argmin()

            if land_mask[lat_idx, lon_idx] < 1:
                return False

            region = land_mask[
                max(0, lat_idx - 1) : min(lat_idx + 2, land_mask.shape[0]),
                max(0, lon_idx - 1) : min(lon_idx + 2, land_mask.shape[1]),
            ]
            return np.all(region == 1)

        for _ in range(1000):
            lat = random.uniform(lat_min, lat_max)
            lon = random.uniform(lon_min, lon_max)
            if is_valid_position(lat, lon):
                return (lat, lon)
            print(f"Invalid position: ({lat}, {lon})")

        raise RuntimeError("Could not find a valid position...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a trained PPO model in a custom environment.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data file.")
    parser.add_argument("--lat", type=float, help="Starting latitude for the model.")
    parser.add_argument("--lon", type=float, help="Starting longitude for the model.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--model_file", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--render_mode", type=str, default="human", choices=["human", "ansi"], help="Rendering mode ('human' or 'ansi').")
    parser.add_argument("--save", action='store_true', help="Save the visualization as an .mp4 file")
    parser.add_argument("--stack", type=int, default=1, help="Number of observation stacks the agent was trained with.")

    args = parser.parse_args()
    if not args.lon or not args.lat:
        print("Either --lat or --lon missing. Randomizing position.")
        lat, lon = randomize_cutter_position(args.data_file)
        fixed_position = False
    else:
        lat = args.lat
        lon = args.lon
        fixed_position = True
        
    visualize_model(
        args.data_file,
        lat,
        lon,
        args.config_file,
        args.model_file,
        args.render_mode,
        args.save,
        args.stack,
        use_fixed_position=fixed_position
    )
