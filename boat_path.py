from simulation.Visualizer import Visualizer
from agent.Cutter import Cutter

data_path = "data/frames/20250224_180246.h5"
c=Cutter(data_path, 30.1, -80.1, "resources/settings.json")
v=Visualizer(data_path)

c.move('N')
c.move('N')
c.move('N')

v._load_trackline(c.path)

v.run(show=True)
