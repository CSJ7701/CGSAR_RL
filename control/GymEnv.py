from os import truncate
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import numpy as np
from .Cutter import Cutter
from simulation.Visualizer import Visualizer

class GymEnv(gym.Env):
    """
    Custom gym environment to model USCG search and rescue.
    """

    def __init__(self, hdf5_path: str, lat: float, lon: float, config_path: str):
        super().__init__()
        self.data_path =hdf5_path
        self.config_path = config_path
        self.cutter = Cutter(hdf5_path, lat, lon, config_path)

        # Define action space (Discrete: 4 possible movements)
        self.action_space = spaces.Discrete(4)

        obs_values = self.cutter.observe().shape[0]
        self.observation_space = spaces.Box(
            low = np.full(obs_values, -np.inf, dtype=np.float64),
            high = np.full(obs_values, np.inf, dtype=np.float64),
            dtype=np.float64
        )

    def _get_heatmap_value(self):
        """
        Determines value of the heatmap cell the cutter is currently in.
        """
        # Row index, latitude
        lat_idx = np.searchsorted(self.cutter.heatmap_latitudes, self.cutter.lat) - 1
        lat_idx = np.clip(lat_idx, 0, self.cutter.heatmap.shape[0] - 1)
        # Column index, longitude
        lon_idx = np.searchsorted(self.cutter.heatmap_longitudes, self.cutter.lon) - 1
        lon_idx = np.clip(lon_idx, 0, self.cutter.heatmap.shape[1] - 1)
        return self.cutter.heatmap[lat_idx, lon_idx]
    
    def reset(self, seed=None, options=None):
        """
        Reset's the environment to its initial state.
        """
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
            
        data_path = self.cutter.data_path
        lat = self.cutter.path["start"][0][0]
        lon = self.cutter.path["start"][0][1]
        config = self.cutter.config.file_path
        self.cutter = Cutter(data_path, lat, lon, config)

        # Random victim index, to select "true" victim.
        self.victim_index = self.np_random.integers(0, len(self.cutter.victim_position)-1)
        
        # Returns (obs, info)
        return self.cutter.observe(), {}

    def reward(self):

        # Reward for being in a heatmap cell, scaled by heatmap value
        x,y = self.cutter._compute_relative_position()
        heatmap_reward = 10 * self.cutter._get_heatmap_value(x,y) # 0-100 in steps of 10

        # Reward based on distance to heatmap
        distance = self.cutter._get_heatmap_distance(x,y)
        distance_reward = 5 * (1/distance) # Inverse scale based on distance

        # Reward for finding a victim
        victim_reward = 0
        if self.cutter.victim_check():
            victim_reward = 1000

        # Penalty for running aground
        aground_penalty = -5000 if self.cutter.is_aground() else 0

        # Penalty for timestep
        if not self.cutter.current_step:
            time_penalty = 0
        else:
            # time_penalty = -1 * int(self.cutter.current_step)
            time_penalty = -1

        # print({
        #     "heatmap": heatmap_reward,
        #     "distance": distance_reward,
        #     "victim": victim_reward,
        #     "aground": aground_penalty,
        #     "time": time_penalty
        # })

        total_reward = heatmap_reward + distance_reward + victim_reward + aground_penalty + time_penalty
        return total_reward

    def step(self, action):
        """
        Executes a step in the environment.

        :param action: An integer representing the direction (0=N, 1=S, 2=E, 3=W).
        :return: observation, reward, terminated, truncated, info
        """
        if self.cutter.current_step >= self.cutter.max_steps:
            truncated = True
            terminated = False
            return self.cutter.observe(), 0, terminated, truncated, {}
        
        direction_map = {0:'N', 1:'S', 2:'E', 3:'W'}
        direction = direction_map[int(action)]
        self.cutter.update(direction)

        obs = self.cutter.observe()
        reward = self.reward()
        
        terminated = self.cutter.is_aground() or self.cutter.victim_check()
        truncated = (self.cutter.current_step >= self.cutter.max_steps-1) and not terminated

        if terminated or truncated:
            #print(f"\033[92m Episode ended at step {self.cutter.current_step} with {'termination' if terminated else 'truncation'}\033[00m.")
            ...

        return obs, reward, terminated, truncated, {}

    def render(self, mode="human", show=False):
        """
        Render the Cutter's state (for debugging).
        """
        if mode == "human":
            v = Visualizer(self.data_path)
            v._load_trackline(self.cutter.path)
            v._load_real_victim(self.cutter.victim_index)
            v.run(show=show)
        elif mode == "ansi":
            print(f"Step: {self.cutter.current_step}/{self.cutter.max_steps} | Cutter Position: Lat={self.cutter.lat}, Lon={self.cutter.lon}")
