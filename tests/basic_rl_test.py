import gymnasium as gym
from control.GymEnv import GymEnv

# Create your environment
env = GymEnv("data/frames/20250224_180246.h5",30.0, -80.1, "resources/settings.json") # Replace with your actual environment class
obs, info = env.reset(seed=42)  # Reset with a fixed seed for reproducibility

# Number of test episodes
num_episodes = 5
max_steps_per_episode = 100

for episode in range(num_episodes):
    obs, info = env.reset()  # Reset env at start of episode
    total_reward = 0
    
    print(f"Episode {episode + 1} starting...")

    for step in range(max_steps_per_episode):
        action = env.action_space.sample()  # Take a random action
        obs, reward, done, truncated, info = env.step(action)
        
        total_reward += reward
        print(f"Step {step+1}: Action={action}, Reward={reward}, Done={done}, Truncated={truncated}")

        if done or truncated:  # End the episode if necessary
            print(f"Episode {episode + 1} finished after {step+1} steps with total reward {total_reward}\n")
            break
