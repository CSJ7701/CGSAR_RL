import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
episodes = 1_000_000
x = np.arange(episodes)

p1_end = 600_000
p2_end = 800_000

p1_end_height = 0.004
p2_end_height = 0.05

# Base reward curve: piecewise linear
base = np.zeros(episodes)
#base[:p1_end] = np.linspace(0, 100, p1_end)
#base[p1_end:p2_end] = np.linspace(100, 300, p2_end - p1_end)
#base[p2_end:] = 300  # plateau

walk = np.zeros(episodes)
#walk[:p1_end] = np.cumsum(np.random.normal(0.005, 1, p1_end))
walk[:] = np.cumsum(np.random.normal(0.005, 3, episodes))
#walk[p1_end:p2_end] = walk[p1_end - 1] + np.cumsum(np.random.normal(0.02, 1.5, p2_end - p1_end))
#walk[p2_end:] = walk[p2_end - 1] + np.cumsum(np.random.normal(0.01, 1.5, episodes - p2_end))

# Add rare large perturbations with lingering effect
perturbations = np.zeros(episodes)
for _ in range(30):
    idx = np.random.randint(50_000, p2_end)
    drop = np.random.uniform(-300, 2000)
    length = 150_000
    x_vals = np.linspace(-3,5,length)
    decay = np.exp(-x_vals**2)
    perturbations[idx:idx+150000] += drop * decay

# Combine all
reward = base + walk + perturbations

# Light smoothing for presentation
def organic_smooth(y, k=300):
    kernel = np.ones(k) / k
    return np.convolve(y, kernel, mode='same')

smoothed = organic_smooth(reward, 10)

# Plot
plt.figure(figsize=(14, 6))
plt.plot(x, smoothed, linewidth=1, label="Reward")

# Phase shading
#plt.axvspan(0, p1_end, color='blue', alpha=0.1, label="Gradual Learning")
#plt.axvspan(p1_end, p2_end, color='orange', alpha=0.1, label="Learning Jump")
#plt.axvspan(p2_end, episodes, color='green', alpha=0.1, label="Plateau")

plt.xlabel("Episode")
plt.ylabel("Reward")
#plt.title("Ideal Reward Curve")
#plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("PRESENTATION_bad_rl_curve.png", dpi=300)
plt.show()
