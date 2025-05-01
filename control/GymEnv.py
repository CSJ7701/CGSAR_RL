import random
from typing import Tuple
from collections import deque
import gymnasium as gym
from gymnasium import spaces
import h5py
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
        self.action_space = spaces.Discrete(5)
        self.action_queue = deque(maxlen=5)

        # obs_values = self.cutter.observe().shape[0]
        # self.observation_space = spaces.Box(
        #     low = np.full(obs_values, -np.inf, dtype=np.float64),
        #     high = np.full(obs_values, np.inf, dtype=np.float64),
        #     dtype=np.float64
        # )

        self.setup_new_observation_space()
        self.fixed_position = None

    def setup_observation_space(self):
        # Calculate the total size of the observation space.
        #window_size = 5
        #local_heatmap_size = (2*window_size + 1) ** 2
        local_heatmap_size = 0
        scalar_features = 9

        obs_size = scalar_features + local_heatmap_size

        self.observation_space = spaces.Box(
            low = np.full(obs_size, -1.0, dtype = np.float32),
            high = np.full(obs_size, 1.0, dtype = np.float32),
            dtype = np.float32
        )

    def setup_new_observation_space(self):
        # Local grid around the cutter
        grid_size = 9
        channels = 5 # depth, current x, current y, heatmap, visibility

        grid_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(grid_size, grid_size, channels),
            dtype=np.float32
        )

        # Vessel state
        # These should be scalars - box, discrete, or just dict entries?
        agent_space = spaces.Box(
            low=np.array([-1.0, 0.0]),
            high=np.array([1.0, 1.0]),
            shape=(2,), # Depth under keel (norm), time step (norm)
            dtype = np.float32
        )

        # Navigation information
        nav_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            shape=(5,), # Direction vectors, normalized distance
            dtype=np.float32
        )

        # Combine into Dict space
        observation_space = spaces.Dict({
            'grid': grid_space,
            'agent': agent_space,
            'nav': nav_space
        })

        self.observation_space = observation_space

        
    def _randomize_cutter_position(self) -> Tuple[float, float]:

        with h5py.File(self.data_path, 'r') as data:
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
                    max(0, lat_idx-1) : min(lat_idx + 2, land_mask.shape[0]),
                    max(0, lon_idx-1) : min(lon_idx + 2, land_mask.shape[1]),
                ]
                return np.all(region==1)

            for _ in range(1000):
                lat = random.uniform(lat_min, lat_max)
                lon = random.uniform(lon_min, lon_max)
                if is_valid_position(lat, lon):
                    return (lat, lon)
                #print(f"Invalid position: ({lat}, {lon})")
            raise RuntimeError("Could not find a valid position...")
                
        

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

    def use_fixed_position(self, lat, lon):
        self.fixed_position = lat, lon
    
    def reset(self, seed=None, options=None):
        """
        Reset's the environment to its initial state.
        """
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
            
        data_path = self.cutter.data_path
        #lat = self.cutter.path["start"][0][0]
        #lon = self.cutter.path["start"][0][1]
        # if options and 'initial_position' in options:
        if self.fixed_position:
            self.cutter_position = self.fixed_position
        else:
            self.cutter_position = self._randomize_cutter_position()
        lat, lon = self.cutter_position
        config = self.cutter.config.file_path
        self.cutter = Cutter(data_path, lat, lon, config)

        # Random victim index, to select "true" victim.
        self.victim_index = self.np_random.integers(0, len(self.cutter.victim_position)-1)
        self.cutter._load_true_victim(self.victim_index)
        
        # Returns (obs, info)
        return self.cutter.observe_dict(), {}

    def _compute_straightness(self):
        """
        Compute a reward based on changes in orientation.
        Encourages efficient search patterns by penalizing excessive turning.
        """
        if len(self.action_queue) < 2:
            return 0 # Not enough data to judge behavior
        non_forward_count = sum(1 for action in self.action_queue if action != 0)

        if non_forward_count == 0:
            return 0.1
        elif non_forward_count <= 2:
            return 0
        else:
            return -0.1

    def reward(self):
        x,y = self.cutter._compute_relative_position()

        # Reward for being in a heatmap cell, scaled by heatmap value
        heatmap_value = self.cutter._get_heatmap_value(self.cutter.lat, self.cutter.lon)
        #heatmap_reward = 1 * heatmap_value # 0-10 scale 
        heatmap_reward = 0

        # # Replace straightness reward curve with static punishment
        #straightness_reward = self._compute_straightness()
        current_action = self.action_queue[-1] if len(self.action_queue) > 0 else 0
        straightness_reward = -0.1 if current_action != 0 else 0

        # Reward based on distance to heatmap - exponential decay
        distance = self.cutter._get_heatmap_distance(x,y)
        #distance_reward = 5 * np.exp(-distance/5.0)
        #distance_reward = -(distance**2) if heatmap_value == 0 else 0
        distance_reward = np.tanh(-distance/15) if heatmap_value == 0 else 5
        # Tanh gives curved falloff from 0 to -1. decent slope until coefficient (15), then starts to level off and meet asymptote

        # Main reward for finding a victim
        victim_reward = 20000 if self.cutter.victim_check() else 0

        # Penalty for running aground
        aground_penalty = -5000 if self.cutter.is_aground() else 0

        # Small time penalty to encourage efficiency
        time_penalty = -1

        total_reward = (
            heatmap_reward +
            distance_reward +
            victim_reward +
            aground_penalty +
            time_penalty +
            straightness_reward
        )
        print(f"HH: {heatmap_value} || S: {straightness_reward} || D: {distance_reward} || T: {total_reward}")
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
            return self.cutter.observe_dict(), 0, terminated, truncated, {}

        # Forward MUST be 0 for the straightness_reward to work correctly
        direction_map = {0:'forward', 1:'forward_left', 2:'forward_right', 3:'left', 4:'right'}
        direction = direction_map[int(action)]
        self.cutter.update(direction)
        self.action_queue.append(int(action))

        obs = self.cutter.observe_dict()
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
            v._load_visibility(self.cutter.mask_history)
            v.run(show=show)
        elif mode == "ansi":
            print(f"Step: {self.cutter.current_step}/{self.cutter.max_steps} | Cutter Position: Lat={self.cutter.lat}, Lon={self.cutter.lon}")
