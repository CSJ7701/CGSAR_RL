import os
from simulation.environment import Environment

lat = 30.0
lon = -80.0
margin = 5 # miles

proj_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(proj_dir,"resources/settings.json")

e = Environment(lat, lon, config_path)
e.PlotNet()
