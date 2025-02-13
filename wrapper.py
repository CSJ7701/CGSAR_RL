import math
import os
from datetime import datetime

from application.logger import Logger
from simulation.Simulation import Simulation
from simulation.Victim import Victim

import random
def generate_victims(config: dict, env, config_path:str):
    def miles_to_degrees(miles):
        radius_of_earth_miles = 3963.0
        degrees_per_mile = 1/radius_of_earth_miles * (180/math.pi)
        return miles * degrees_per_mile
    
    N = config.get('N',10)
    lat = config.get('lat', 30.1)
    lon = config.get('lon', -80.0)
    range_miles = config.get('range_miles', 2)

    victims=[]
    lat_range = miles_to_degrees(range_miles)
    lon_range = miles_to_degrees(range_miles)

    for i in range(N):
        lat_p = lat+random.uniform(-lat_range, lat_range) if config.get('perturb_lat', False) else lat
        lon_p = lon+random.uniform(-lon_range, lon_range) if config.get('perturb_lon', False) else lon

        xy_p = random.uniform(config.get('xy_min', 0.2), config.get('xy_max', 2)) if config.get('perturb_xy', False) else 0.5
        z_p = random.uniform(config.get('z_min', 0.5), config.get('z_max', 2)) if config.get('perturb_z', False) else 1

        victim_type = random.choice(config.get('victim_types', ['piw', 'piw_lj'])) if config.get('perturb_victim_type', False) else 'piw'

        victim = Victim(xy_p, xy_p, z_p, lat_p, lon_p, victim_type, env, config_path, i+1)

        victims.append(victim)
        print(f"Number of victims: {i+1}, Length of victim array: {len(victims)}")
    return victims

# Victim Config
config = {
    'N': 2,  # Number of victims to generate
    'lat': 30.1,  # Latitude
    'lon': -80.0,  # Longitude
    'range_miles': 2,  # Range for perturbations in miles
    'x_min': 1,  # Minimum x value for perturbation
    'x_max': 15,  # Maximum x value for perturbation
    'y_min': 1,  # Minimum y value for perturbation
    'y_max': 15,  # Maximum y value for perturbation
    'z_min': 5,  # Minimum z value for perturbation
    'z_max': 30,  # Maximum z value for perturbation
    'victim_types': ['piw', 'piw_lj'],  # Victim types to randomly select from
    'perturb_lat': True,  # Whether to perturb latitude
    'perturb_lon': True,  # Whether to perturb longitude
    'perturb_x': False,  # Whether to perturb x axis
    'perturb_y': False,  # Whether to perturb y axis
    'perturb_z': False,  # Whether to perturb z axis
    'perturb_victim_type': False,  # Whether to perturb victim type
}


# Static variables. These should not change.
logger = Logger(__name__).get()
proj_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(proj_dir,"resources/settings.json")

# Tuning parameters.
# These can change, and will likely be determined by some 'randomizer' later.
lat = 30.1
lon = -80.0
start_date = datetime(2023, 1, 1, 00, 00, 00)
end_date = datetime(2023, 1, 1, 00,30,00)

logger.info({"event": "simulation_start", "message": "Starting Simulation", "data": {"Center": (lat, lon), "StartDate": str(start_date.isoformat()), "EndData": str(end_date.isoformat())}})

s = Simulation(lat, lon, config_path, start_date, end_date)
#v1=Victim(10,10,10, 30.1, -80.0, "piw", s.env, config_path, 1)
#v2=Victim(0.5,0.5,3, 30.3, -80.0, "piw", s.env, config_path, 2)
#s._add_victim(v1)
#s._add_victim(v2)

victims = generate_victims(config, s.env, config_path)
for v in victims:
    s._add_victim(v)

s.Run(file='test.mp4')

#mps = v.Displacement()/((end_date-start_date).total_seconds())
#print(f"START: {v.start} --- END: {v.position} --- Displacement: {v.Displacement()} --- Avg Vel: {mps}")



