
from control.GymEnv import GymEnv
from simulation.Visualizer import Visualizer


v = Visualizer("data/simulations/Apr19160104/env_1/vics_1/20250419_160105.h5")
v.victim_positions = None
v.run(show=True)
