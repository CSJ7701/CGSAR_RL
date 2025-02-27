import numpy as np
from control.GymEnv import GymEnv
from gymnasium.utils.env_checker import check_env

def test_environment():
    print("Starting environment tests...\n")
    
    # 1. Smoke Test: Create and Reset the Environment
    print("[1] Initializing environment...")
    try:
        env = GymEnv("/home/csj7701/Projects/CGSAR_RL/data/frames/20250224_180246.h5", 40.0, -70.0, "config.json")
        print("✅ Environment initialized successfully.")
    except Exception as e:
        print(f"❌ Environment initialization failed: {e}")
        return
    
    print("[2] Resetting environment...")
    try:
        obs,_ = env.reset()
        print("✅ Environment reset successfully.")
    except Exception as e:
        print(f"❌ Environment reset failed: {e}")
        return
    
    assert isinstance(obs, np.ndarray), f"❌ Observation is not a NumPy array, is {type(obs)}"
    assert obs.shape == env.observation_space.shape, "❌ Observation shape mismatch"
    print("✅ Observation is a valid NumPy array with correct shape.\n")
    
    # 2. Step Test: Take a Few Random Steps
    print("[3] Taking 10 random steps...")
    try:
        for i in range(10):
            action = env.action_space.sample()
            print(f"Taking action: {action}")
            obs, reward, done, truncated, _ = env.step(action)
            print(f"Step {i+1}: Action={action}, Reward={reward}, Done={done}, Truncated={truncated}")
            if done or truncated:
                print("✅ Episode ended early, as expected.")
                break
        print("✅ Random steps executed successfully.\n")
    except Exception as e:
        print(f"❌ Error during step execution: {e}")
        return
    
    # 3. Heatmap Value Check
    print("[4] Checking heatmap reward calculation...")
    try:
        heatmap_value = env._get_heatmap_value()
        assert isinstance(heatmap_value, (float, np.float32, np.float64)), "❌ Heatmap value is not a float"
        print(f"✅ Heatmap value retrieved successfully: {heatmap_value}\n")
    except Exception as e:
        print(f"❌ Heatmap reward calculation failed: {e}")
        return
    
    # 4. Run Full Episode Test
    print("[5] Running a full episode test...")
    try:
        obs, _ = env.reset(), False
        while True:
            action = env.action_space.sample()
            obs, reward, done, truncated, _ = env.step(action)
            if done or truncated:
                print("✅ Episode terminated correctly.")
                break
        print("✅ Full episode test completed successfully.\n")
    except Exception as e:
        print(f"❌ Full episode test failed: {e}")
        return
    
    # 5. Gym Compatibility Test
    print("[6] Running Gym environment compatibility test...")
    try:
        check_env(env)
        print("✅ Environment is Gym-compatible.\n")
    except Exception as e:
        print(f"❌ Gym compatibility check failed: {e}")
        return
    
    print("All tests passed successfully! 🎉")

if __name__ == "__main__":
    test_environment()
