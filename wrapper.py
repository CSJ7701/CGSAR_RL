import math
import os
import random
import numpy as np
from datetime import datetime

from application.logger import Logger
from simulation.VictimGroup import VictimGroup
from simulation.Simulation import Simulation
from simulation.Visualizer import Visualizer

# Static variables. These should not change.
logger = Logger(__name__).get()
proj_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(proj_dir,"resources/settings.json")

# Tuning parameters.
# These can change, and will likely be determined by some 'randomizer' later.
lat = 30.1
lon = -80.0
start_date = datetime(2023, 1, 1, 00, 00, 00)
end_date = datetime(2023, 1, 2, 00,00,00)

logger.info({"event": "simulation_start", "message": "Starting Simulation", "data": {"Center": (lat, lon), "StartDate": str(start_date.isoformat()), "EndData": str(end_date.isoformat())}})

s = Simulation(lat, lon, config_path, start_date, end_date)

num_victims=2000
lats=30.3 + np.random.uniform(-0.05, 0.05, num_victims)
lons=-80.0 + np.random.uniform(-0.05, 0.05, num_victims)
x=np.full(num_victims, 0.5)
y=np.full(num_victims, 0.5)
z=np.full(num_victims, 1)
v_type=np.full(num_victims, "piw")
vics = VictimGroup(x=x, y=y, z=z, lat=lats, lon=lons, victim_type=v_type, env=s.env, config_path=config_path)
s._add_victim_group(vics)

s.Run()
#vis = Visualizer(s.hdf5_path)
#vis.run()




