
from simulation.Visualizer import Visualizer
from control.Cutter import Cutter
import h5py

data_path = "data/frames/env_no_vics.h5"
initial_step = 1
c = Cutter(data_path, 30.0, -80.1, "resources/settings.json", initial_step=initial_step)
v = Visualizer(data_path)

c.orientation = 0
c.update('forward')
c.orientation = 180
c.update('forward')
c.update('forward')
c.orientation = 0
c.update('forward')
c.orientation = 90
c.update('forward')
c.orientation = 270
c.update('forward')
c.update('forward')
for _ in range(20):
    c.orientation = 90
    c.update('forward')
    c.update('forward')
    c.orientation = 270
    c.update('forward')
    c.update('forward')

v._load_trackline(c.path)
v.run(show=True)
