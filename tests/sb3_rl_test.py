import gymnasium as gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from control.GymEnv import GymEnv
from math import ceil

# ✅ Initialize your custom environment
env = GymEnv("data/frames/big_env_2.h5",30.0, -80.1, "resources/settings.json")
env_vis_mode = "human"

# ✅ Create the PPO model
model = PPO(
   "MlpPolicy",  # Multi-layer perceptron policy (MLP) for standard observation spaces
    env,
    verbose=1,     # Print training info
    device="cpu",
    tensorboard_log="./data/tensorboard",  # Log training data for TensorBoard
    n_steps = env.cutter.max_steps,
    batch_size = 72,
)


# ✅ Train the model
print("Starting training...")
# Makes sure I can't screw up episode size and accidentally train for 14 hours... (We learn from our mistakes)
episode_size = env.cutter.max_steps
target_episodes = 100000
episode_scale_factor = ceil(target_episodes / episode_size)
model.learn(total_timesteps=episode_scale_factor * episode_size)  # Train for 100 episodes
print("Training complete!")

# ✅ Save the trained model
model.save("ppo_mymodel")

# ✅ Load the trained model (optional)
model = PPO.load("ppo_mymodel", env)

# ✅ Run a test episode
obs, _ = env.reset()
done = False
truncated = False
while not done and not truncated:
    action, _states = model.predict(obs)  # Get action from trained policy
    obs, reward, done, truncated, _ = env.step(action)
    if env_vis_mode == "ansi":
        env.render(mode="ansi")  # Render if your env supports it
if env_vis_mode == "human":
    env.render(mode="human")

env.close()
