import gymnasium as gym
import numpy as np
from collections import deque
from gymnasium.spaces import Dict,Box
from typing import Dict as TypeDict, Optional, Union, Tuple, Any

class ObservationStackingWrapper(gym.Wrapper):
    """
    Wrapper for stacking observations over time.
    Works with dictionary observation spaces by stacking each numeric component.
    """

    def __init__(self, env: gym.Env, num_stack: int = 4):
        """
        Initialize the wrapper with the environment to wrap and the number of frames to stack.

        :param env: The environment to wrap.
        :param num_stack: Number of observations to stack.
        """
        super().__init__(env)
        self.num_stack = num_stack
        self.frames = None

        # Modify the observation space to accomodate stacked frames
        if isinstance(self.observation_space, Dict):
            spaces = {}

            for key, space in self.observation_space.spaces.items():
                if isinstance(space, Box):
                    # For Box spaces, stack along a new axis at the end
                    if len(space.shape) == 1: # 1D array - stack as an additional dimension
                        low = np.repeat(space.low[...,np.newaxis], self.num_stack, axis=-1)
                        high = np.repeat(space.high[...,np.newaxis], self.num_stack, axis=-1)
                        spaces[key] = Box(low=low, high=high, dtype=space.dtype)
                    elif len(space.shape) >= 2: # For multidimensional arrays (like grid)
                        # Get the original shape and expand the last dimension
                        original_shape = space.shape
                        # For a 3D tensor like (9,9,5) we want to make it (9,9,5*num_stack)
                        new_shape = list(original_shape[:-1]) + [original_shape[-1] * self.num_stack]

                        # Create the new box with expanded last dimension
                        spaces[key] = Box(
                            low = np.tile(space.low, [1] * (len(original_shape)-1) + [self.num_stack]),
                            high = np.tile(space.high, [1] * (len(original_shape)-1) + [self.num_stack]),
                            shape = tuple(new_shape),
                            dtype = space.dtype
                        )
                else:
                    # For non box spaces, don't stack
                    spaces[key] = space

            self.observation_space = Dict(spaces)
        else:
            raise ValueError("This wrapper only works with Dict observation spaces.")

    def _get_observation(self) -> TypeDict[str, np.ndarray]:
        """
        Stack the observations in the deque and return them as a stacked observation.

        :return: Dictionary containing stacked observations.
        """
        assert self.frames is not None, "Frames deque is not initialized."

        stacked_obs = {}
        for key in self.frames[0].keys():
            if isinstance(self.observation_space.spaces[key], Box):
                # Stack only box spaces
                if len(self.frames[0][key].shape) == 1: # 1D array
                    stacked_obs[key] = np.stack([f[key] for f in self.frames], axis=-1)
                else: # Multi-dimensional array, like grid
                    # For multi-dimensional arrays like (9,9,5) concatenate along the last axis
                    stacked_frames = [f[key] for f in self.frames]
                    stacked_obs[key] = np.concatenate(stacked_frames, axis=-1)
            else:
                # For non-box spaces, just use latest observation
                stacked_obs[key] = self.frames[-1][key]

        return stacked_obs

    def reset(self, **kwargs) -> Tuple[TypeDict[str, np.ndarray], dict]:
        """
        Reset the environment and initialize the frame buffer with the initial observation.

        :returns: Stacked initial observation and info dict.
        """
        observation, info = self.env.reset(**kwargs)

        # Initialize the frames deque with copies of the initial observation
        self.frames = deque([observation] * self.num_stack, maxlen=self.num_stack)
        return self._get_observation(), info

    def step(self, action: Any) -> Tuple[TypeDict[str, np.ndarray], float, bool, bool, dict]:
        """
        Step the environment and update the frame buffer with the new observation.

        :param arg: The action to take.

        :returns: Stacked observation, reward, terminated, truncated, and info.
        """
        observation, reward, terminated, truncated, info = self.env.step(action)

        # Update the frame buffer
        self.frames.append(observation)

        return self._get_observation(), reward, terminated, truncated, info
                    
        
