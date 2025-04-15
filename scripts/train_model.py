from math import ceil
import argparse
import os
import random
import subprocess
import re
import h5py
import numpy as np
from typing import Optional, Tuple
import gymnasium as gym
import torch
from tqdm import tqdm
from stable_baselines3 import PPO

from control.GymEnv import GymEnv
from application.logger import Logger

logger = Logger("Train").get()

def parse_args():
    parser = argparse.ArgumentParser(description="Train an RL model repeatedly on a set of pre-generated simulation environments.")

    train = parser.add_argument_group("Training Parameters", "Static parameters that determine the structure of the model's training.")
    train.add_argument("-S", "--simulations", type=int, required=True, help="Number of simulation environments to train on.")
    train.add_argument("-I", "--iterations", type=int, required=True, help="Number of iterations per environment.")
    train.add_argument("-E", "--episodes", type=int, required=True, help="Number of episodes per iteration.")
    train.add_argument("-e", "--environment_dir", type=str, required=True, help="Path to the simulation data directory, or single file.")
    
    train.add_argument("-y", "--evaluation", type=int, default=None, help="Interval in iterations at which to show an evaluation step. Default is 0, or no evaluation. If enabled, higher numbers are recommended.")

    cutter = parser.add_argument_group("Agent Parameters", "Parameters to define the behavior of the agent, representing a CG Cutter.")
    cutter.add_argument("-p", "--position", type=float, nargs=2, default=None, help="Specify a fixed position (lat, lon)")
    

    return parser.parse_args()


def validate_data_dir(directory: str) -> bool:
    """
    Validate if the given directory matches the expected structure:
    - Multiple folders named "env_X" where X is a number.
    - Each "env_X" contains multiple folders named "vics_Y" where Y is a number.
    - Each "vics_Y" folder contains at least one ".h5" file.
    :param directory: Path to the directory.
    :return: True if the structure matches, False otherwise.
    """
    if not os.path.isdir(directory):
        if directory.endswith(".h5"):
            return True
        logger.fatal({"event": "invalid_data_dir", "message": "Data directory does not exist."})
        return False

    env_pattern = re.compile(r"env_\d+")
    vic_pattern = re.compile(r"vics_\d+")

    env_folders = [f for f in os.listdir(directory) if env_pattern.fullmatch(f) and os.path.isdir(os.path.join(directory, f))]
    if not env_folders:
        logger.warning({"event": "invalid_data_dir", "message": "No environment folders found."})
        return False

    for env in env_folders:
        env_path = os.path.join(directory, env)
        vics_folders = [f for f in os.listdir(env_path) if vic_pattern.fullmatch(f) and os.path.isdir(os.path.join(env_path, f))]
        if not vics_folders:
            logger.warning({"event": "invalid_data_dir", "message": f"No victim folders found under {env}."})
            return False

        for vics in vics_folders:
            vics_path = os.path.join(env_path, vics)
            h5_files = [f for f in os.listdir(vics_path) if f.endswith(".h5")]
            if not h5_files:
                logger.warning({"event": "invalid_data_dir", "message": f"No .h5 files found in {vics} under {env}."})
                return False

    return True

def get_data_file(directory: str) -> str:
    """
    Return a random data file from the given directory.
    Assumes the given directory contains folders name "env_X", where X is a number,
    and that those folders contain other folders named "vics_Y" where Y is a number.
    Will return a file with the extension ".h5" from a random "vics_Y" folder in a random "env_X" folder.
    :param directory: Path to the directory.
    :return: The string path of the random data file. Empty string if none found.
    """

    env_pattern = re.compile(r"env_\d+")
    vic_pattern = re.compile(r"vics_\d+")

    env_folders = [f for f in os.listdir(directory) if env_pattern.fullmatch(f) and os.path.isdir(os.path.join(directory, f))]
    env = random.choice(env_folders)
    env_path = os.path.join(directory, env)
    vic_folders = [f for f in os.listdir(env_path) if vic_pattern.fullmatch(f) and os.path.isdir(os.path.join(env_path,f))]
    vic = random.choice(vic_folders)
    vic_path = os.path.join(env_path, vic)
    data_files = [f for f in os.listdir(vic_path) if f.endswith(".h5")]
    file = random.choice(data_files)
    file_path = os.path.join(vic_path,file)
    return file_path

def randomize_cutter_position(data_file: str) -> Tuple[float, float]:
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

def start_tensorboard(log_dir="./data/tensorboard", port=6006):
    subprocess.Popen(["tensorboard", "--logdir", log_dir, "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def train(data_file: str, episodes: int, cutter_lat: float, cutter_lon: float, model_path: str, tensorboard_path: str = "./data/tensorboard", iteration_bar: Optional[tqdm] = None):

    if not data_file or not episodes:
        raise ValueError("Missing training parameters.")

    env = GymEnv(data_file, cutter_lat, cutter_lon, "resources/settings.json")

    start_tensorboard(log_dir=tensorboard_path)
    
    if not os.path.exists(model_path):
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            device="cuda",
            tensorboard_log=tensorboard_path,
            n_steps = env.cutter.max_steps,
            batch_size = 72
            )
    else:
        model = PPO.load(
            model_path,
            env
            )

    episode_size = env.cutter.max_steps
    target_episodes = episodes
    episode_scale_factor = ceil(target_episodes / episode_size)

    model.learn(total_timesteps = episode_scale_factor * episode_size)

    model.save(model_path)
    if iteration_bar:
        iteration_bar.update(1)

def evaluate(data_file: str, cutter_lat: float, cutter_lon: float, model_path: str):
    if not os.path.exists(model_path):
        raise ValueError("Model does not exist.")
    
    env = GymEnv(data_file, cutter_lat, cutter_lon, "resources/settings.json")
    model = PPO.load(model_path, env)

    obs, _ = env.reset()
    done = False
    truncated = False
    while not done and not truncated:
        action, _states = model.predict(obs)
        obs, reward, done, truncated, _ = env.step(action)

    env.render(mode="human", show=True)

def main():
    args = parse_args()
    total_iterations = 0
    
    if not validate_data_dir(args.environment_dir):
        raise ValueError("Directory structure is not valid.")

    total_iteration_count = args.simulations * args.iterations
    with tqdm(total = total_iteration_count, desc="Total Progress") as total_bar:
        with tqdm(total = args.simulations, desc="Simulations") as sim_bar:
            for sim in range(args.simulations):
                if args.environment_dir.endswith(".h5"):
                    data = args.environment_dir
                else:
                    data = get_data_file(args.environment_dir)
                logger.info({"message": f"Training on env [{sim+1}/{args.simulations}] at {data}"})
                with tqdm(total=args.iterations, desc=f"Iterations (Sim {sim+1})", leave=False) as iteration_bar:
                    for i in range(args.iterations):
                        logger.info({"message": f"Beginning training iteration [{i+1}/{args.iterations}]"})
                        total_iterations+=1
                        if args.position:
                            position = args.position
                        else:
                            position = randomize_cutter_position(data) # lat, lon
                        train(data, args.episodes, position[0], position[1], "first_model", iteration_bar=iteration_bar)
                        if args.evaluation and total_iterations%args.evaluation == 0:
                            evaluate(data, 30.0, -80.1, "first_model")

                        total_bar.update(1)
                sim_bar.update(1)
                
        
    


if __name__ == "__main__":
    main()
