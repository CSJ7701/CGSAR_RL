from simulation.Visualizer import Visualizer
from control.Cutter import Cutter
import h5py

data_path = "data/frames/env_w_vics.h5"
initial_step=1
c=Cutter(data_path, 30.0, -80.1, "resources/settings.json", initial_step=initial_step)
c._load_true_victim(2)
c.orientation = 90

v=Visualizer(data_path)
v._load_real_victim(2)

with h5py.File(data_path, 'r') as data:
    steps = len(data.keys())

#directions = ['N', 'E', 'S', 'W']
#directions = ['N', 'N', 'N', 'E']
directions = ['forward', 'forward', 'forward', 'forward', 'right']
direction_idx = 0
step_count = 0
step_limit = 1
change_count = 0
for step in range(initial_step,steps-1):
    c.update(directions[direction_idx])
    step_count += 1
    
    if step_count == step_limit:
        direction_idx = (direction_idx + 1) % 4
        step_count = 0
        change_count += 1

        if change_count % 2 == 0:
            step_limit += 1

v._load_trackline(c.path)
v._load_visibility(c.mask_history)

v.run(show=True)
