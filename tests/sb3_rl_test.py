import gymnasium as gym
import torch
from stable_baselines3 import PPO
from control.GymEnv import GymEnv

# ✅ Initialize your custom environment
env = GymEnv("data/frames/env_w_vics.h5",30.0, -80.1, "resources/settings.json")
env_vis_mode = "human"

# ✅ Create the PPO model
model = PPO(
    "MlpPolicy",  # Multi-layer perceptron policy (MLP) for standard observation spaces
    env,
    verbose=1,     # Print training info
    device="cpu",
    tensorboard_log="./data/tensorboard",  # Log training data for TensorBoard
)

# ✅ Train the model
print("Starting training...")
model.learn(total_timesteps=100_000)  # Train for 10,000 steps
print("Training complete!")

# ✅ Save the trained model
model.save("ppo_mymodel")

# ✅ Load the trained model (optional)
model = PPO.load("ppo_mymodel", env=env)

# ✅ Run a test episode
obs, _ = env.reset()
done = False
while not done:
    action, _states = model.predict(obs)  # Get action from trained policy
    obs, reward, done, truncated, _ = env.step(action)
    if env_vis_mode == "ansi":
        env.render(mode="ansi")  # Render if your env supports it
if env_vis_mode == "human":
    env.render(mode="human")

env.close()
