# Test script for the updated Cutter observation space
from simulation.Visualizer import Visualizer
from control.Cutter import Cutter
import h5py
import numpy as np
import matplotlib.pyplot as plt

data_path = "data/frames/env_w_vics.h5"
initial_step = 1

print("Initializing Cutter and Vis...")
c = Cutter(data_path, 30.2, -80.0, "resources/settings.json", initial_step=initial_step)
c._load_true_victim(2)

v = Visualizer(data_path)
v._load_real_victim(2)

with h5py.File(data_path, 'r') as data:
    steps = len(data.keys())

directions = ['N', 'N', 'N', 'N']

fig, axes = plt.subplots(2,2,figsize=(15,10))
axes = axes.flatten()

for i in range(4):
    x,y = c._compute_relative_position()
    print(f"\nStep {c.current_step}: Position ({x:.2f}, {y:.2f})")

    obs, local_heatmap = c.observe()

    # Print observation details
    print(f"Depth under keel: {c._get_depth() - c.draft:.2f}m")
    print(f"Victim nearby: {c.victim_check()}")
    print(f"Heatmap value at position: {c._get_heatmap_value(x, y):.4f}")
    #print("\n\n",local_heatmap)
    print(obs)

    # Plot local heatmap
    im = axes[i].imshow(local_heatmap, cmap='hot', interpolation='nearest')
    axes[i].set_title(f"Local Heatmap {i} After Moving {directions[i % 4]}")
    axes[i].set_xlabel("Longitude")
    axes[i].set_ylabel("Latitude")

    print(f"Obs shape: {obs.shape}")

    if i < 3:
        print(f"Moving {directions[i%4]}...")
        c.update(directions[i%4])

fig.colorbar(im, ax=axes, label="Normalized heatmap value")

#plt.tight_layout()
plt.savefig('observation_test_results.png')
#plt.show()

print("\nTest completed. Check plots for visualization of local heatmap and gradients.")

v._load_trackline(c.path)
v.run(show=True)
