from math import ceil
import argparse
import os
import random
import re
import gymnasium as gym
import torch
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
    train.add_argument("-e", "--environment_dir", type=str, required=True, help="Path to the simulation data directory.")
    train.add_argument("-y", "--evaluation", type=int, default=0, help="Interval in iterations at which to show an evaluation step. Default is 0, or no evaluation. If enabled, higher numbers are recommended.")

    
    cutter = parser.add_argument_group("Agent Parameters", "Parameters to define the behavior of the agent, representing a CG Cutter.")
    ...

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
            
def main():
    args = parse_args()
    
    if not validate_data_dir(args.environment_dir):
        raise ValueError("Directory structure is not valid.")
    print(get_data_file(args.environment_dir))


if __name__ == "__main__":
    main()
