import argparse
import gymnasium as gym
import torch
from stable_baselines3 import PPO
from control.GymEnv import GymEnv

def visualize_model(data_file, lat, lon, config_file, model_file, render_mode):
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
            env.render(mode="ansi")
    if render_mode == "human":
        env.render(mode="human", show=True)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a trained PPO model in a custom environment.")
    parser.add_argument("--data_file", type=str, required=True, help="Path to the data file.")
    parser.add_argument("--lat", type=float, required=True, help="Starting latitude for the model.")
    parser.add_argument("--lon", type=float, required=True, help="Starting longitude fo.")
    parser.add_argument("--config_file", type=str, required=True, help="Path to the configuration file.")
    parser.add_argument("--model_file", type=str, required=True, help="Path to the trained model file.")
    parser.add_argument("--render_mode", type=str, default="human", choices=["human", "ansi"], help="Rendering mode ('human' or 'ansi').")

    args = parser.parse_args()

    visualize_model(
        args.data_file,
        args.lat,
        args.lon,
        args.config_file,
        args.model_file,
        args.render_mode
    )
