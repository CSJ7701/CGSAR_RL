def debug_observation_stacking(data_file, lat, lon, config_path="resources/settings.json", num_stack=4):
    """
    Debug tool to examine observation stacking more closely, printing detailed
    shape information at each step.
    """
    from control.GymEnv import GymEnv
    from control.ObservationStackingWrapper import ObservationStackingWrapper
    import numpy as np
    
    # Create environments
    print("Creating base environment...")
    base_env = GymEnv(data_file, lat, lon, config_path)
    print("Base environment created.")
    
    # Print original observation space details
    print("\n=== Original Observation Space ===")
    for key, space in base_env.observation_space.spaces.items():
        print(f"{key}: {space}")
    
    # Reset base environment and print observation shapes
    print("\n=== Original Observation Shapes ===")
    base_obs, _ = base_env.reset()
    for key, value in base_obs.items():
        print(f"{key}: {type(value).__name__} with shape {np.shape(value)}")
    
    # Create stacked environment
    print("\n=== Creating Stacked Environment ===")
    try:
        stacked_env = ObservationStackingWrapper(base_env, num_stack=num_stack)
        print("Stacked environment created successfully!")
        
        # Print stacked observation space details
        print("\n=== Stacked Observation Space ===")
        for key, space in stacked_env.observation_space.spaces.items():
            print(f"{key}: {space}")
        
        # Reset stacked environment and print observation shapes
        print("\n=== Stacked Observation Shapes ===")
        stacked_obs, _ = stacked_env.reset()
        for key, value in stacked_obs.items():
            print(f"{key}: {type(value).__name__} with shape {np.shape(value)}")
        
        # Take a few steps to verify
        print("\n=== Testing Steps ===")
        for i, action in enumerate([0, 1, 2]):
            print(f"\nTaking action {action}...")
            stacked_obs, reward, done, truncated, info = stacked_env.step(action)
            print(f"Step {i+1} successful. Observation shapes:")
            for key, value in stacked_obs.items():
                if isinstance(value, np.ndarray):
                    print(f"  {key}: {np.shape(value)}")
        
        print("\nObservation stacking is working correctly!")
        
    except Exception as e:
        print("\n=== ERROR ===")
        print(f"Error creating or using stacked environment: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Example usage:
    debug_observation_stacking("data/frames/env_w_vics.h5", 30.0, -80.1, num_stack=4)
