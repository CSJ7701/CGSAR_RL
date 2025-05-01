import numpy as np
from control.GymEnv import GymEnv
from control.ObservationStackingWrapper import ObservationStackingWrapper

def visualize_observation_stacking(data_file, lat, lon, config_path="resources/settings.json", num_stack=4):
    """
    Visualize how observation stacking changes the observation space and actual observations.
    """
    # Create the base environment
    base_env = GymEnv(data_file, lat, lon, config_path)
    
    # Create the stacked environment
    stacked_env = ObservationStackingWrapper(base_env, num_stack=num_stack)
    
    # Reset both environments
    base_obs, _ = base_env.reset()
    stacked_obs, _ = stacked_env.reset()
    
    # Print observation space information
    print("\n===== OBSERVATION SPACE COMPARISON =====")
    print(f"Base environment observation space:")
    for key, space in base_env.observation_space.spaces.items():
        print(f"  {key}: {space}")
    
    print(f"\nStacked environment observation space:")
    for key, space in stacked_env.observation_space.spaces.items():
        print(f"  {key}: {space}")
    
    # Print actual observation shapes
    print("\n===== OBSERVATION SHAPES COMPARISON =====")
    print(f"Base observation shapes:")
    for key, obs in base_obs.items():
        print(f"  {key}: {type(obs).__name__} with shape {np.shape(obs)}")
    
    print(f"\nStacked observation shapes:")
    for key, obs in stacked_obs.items():
        print(f"  {key}: {type(obs).__name__} with shape {np.shape(obs)}")
    
    # Take a few actions to see how stacking evolves
    print("\n===== TAKING ACTIONS TO SEE STACKING EVOLUTION =====")
    actions = [0, 1, 2, 3, 0]  # Some example actions
    
    for i, action in enumerate(actions):
        # Step both environments
        _, _, _, _, _ = base_env.step(action)
        stacked_obs, _, _, _, _ = stacked_env.step(action)
        
        print(f"\nAfter action {i+1} ({action}):")
        for key, obs in stacked_obs.items():
            if isinstance(obs, np.ndarray):
                print(f"  {key}: shape {np.shape(obs)}")
                
                # For smaller arrays, we can show the actual values
                if key == 'agent' and len(obs.shape) <= 2:
                    print(f"    Values: {obs.flatten()}")

if __name__ == "__main__":
    # Example usage:
    visualize_observation_stacking("data/frames/env_w_vics.h5", 30.0, -80.1)
    # pass
