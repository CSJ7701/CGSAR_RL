import os
from datetime import datetime
from simulation.Simulation import Simulation
from simulation.environment import Environment

lat = 30.0
lon = -80.0
margin = 5 # miles

proj_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(proj_dir,"resources/settings.json")
start_date = datetime(2023, 1, 1, 00, 00, 00)
end_date = datetime(2023, 1, 5, 00,00,00)

e = Environment(lat, lon, config_path)
#e.Plot()

s = Simulation(lat, lon, config_path, start_date, end_date)
s.Run()
