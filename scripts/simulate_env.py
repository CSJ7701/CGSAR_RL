import os
import numpy as np
from datetime import datetime
import argparse

from application.logger import Logger
import logging
from simulation.VictimGroup import VictimGroup
from simulation.Simulation import Simulation

# Static Variables
logger = Logger(__name__).get()
logger.setLevel(logging.INFO)
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(proj_dir,"resources/settings.json")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate multiple simulation environments with varying victim sets.")

    parser.add_argument("-E", "--env", type=int, default=1, help="Number of environments to generate")
    parser.add_argument("-V", "--vic", type=int, default=5, help="Number of victim groups per environment")
    parser.add_argument("-C", "--count", type=int, default=2, help="Number of iterations for each victim_group")
    parser.add_argument("-v", "--num_victims", type=int, default=2000, help="Number of victims per group")

    # Currently, simulation center and victim placement are the same.
    parser.add_argument("--lat", type=float, default=30.1, help="Base latitude for simulation")
    parser.add_argument("--lon", type=float, default=-80.0, help="Base longitude for simulation")
    parser.add_argument("--lat_delta", type=float, nargs=2, default=[-0.05, 0.05], help="Latitude variation range (low high)")
    parser.add_argument("--lon_delta", type=float, nargs=2, default=[-0.05, 0.05], help="Longitude variation range (low high)")

    parser.add_argument("-s", "--start_date", type=str, default="2023-01-01T00:00:00", help="Simulation start date (YYYY-MM-DDTHH:MM:SS)")
    parser.add_argument("-e", "--end_date", type=str, default="2023-01-02T00:00:00", help="Simulation end date (YYYY-MM-DDTHH:MM:SS)")
    parser.add_argument("-o", "--output_dir", type=str, default=os.path.join(proj_dir, "data/simulations"), help="Directory to store simulation outputs")

    return parser.parse_args()

def generate_dateseed():
    return datetime.now().strftime("%b%d%H%M%S")

def main():
    args = parse_args()

    # Convert date strings to datetime objects
    start_date = datetime.strptime(args.start_date, "%Y-%m-%dT%H:%M:%S")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%dT%H:%M:%S")

    dateseed = generate_dateseed()
    base_output_dir = str(os.path.join(args.output_dir, dateseed))
    os.makedirs(base_output_dir, exist_ok=True)
    logger.info({"message": f"Saving simulations to base directory {base_output_dir}", "event": "begin_simulation_loop"})
    for i in range(args.env): # Generate 'E' environments
        env_dir = os.path.join(base_output_dir, f"env_{i+1}")
        logger.info({"message": f"Initializing environment {i+1}", "event": "simulation_loop_environment", "data":{"env_dir": env_dir}})
        os.makedirs(env_dir, exist_ok=False)

        # Create the environment
        sim = Simulation(args.lat, args.lon, config_path, start_date, end_date)

        for j in range(args.vic): # Run 'V' victim groups per environment
            vic_dir = os.path.join(env_dir, f"vics_{j+1}")
            logger.info({"message": f"Initializing victim group {j+1}", "event": "simulation_loop_victim", "data":{"vic_dir": vic_dir}})
            os.makedirs(vic_dir, exist_ok=False)

            # Generate victim locations
            lats = args.lat + np.random.uniform(args.lat_delta[0], args.lat_delta[1], args.num_victims)
            lons = args.lon + np.random.uniform(args.lon_delta[0], args.lon_delta[1], args.num_victims)
            v_type = np.full(args.num_victims, "piw")

            # Placeholder for x,y,z (until VictimGroup is refactored)
            x = np.full(args.num_victims, 0.5)
            y = np.full(args.num_victims, 0.5)
            z = np.full(args.num_victims, 1.0)

            vics = VictimGroup(x=x, y=y, z=z, lat=lats, lon=lons, victim_type=v_type, env=sim.env, config_path=config_path)
            sim._add_victim_group(vics)

            # Run with output directory
            for c in range(args.count):
                sim.Run(save_dir=vic_dir)

def example():
    # Tuning parameters
    # Should be able to pass in as args.
    simulation_center_lat = 30.1
    simulation_center_lon = -80.0

    num_victims = 2000

    victim_lat = 30.1
    victim_lat_delta_low = -0.05
    victim_lat_delta_high = 0.05

    victim_lon = -80.0
    victim_lon_delta_low = -0.05
    victim_lon_delta_high = 0.05

    start_date = datetime(2023,1,1,00,00,00)
    end_date = datetime(2023,1,2,00,00,00)

    # Semi-Random Initial Conditions
    lats = victim_lat + np.random.uniform(victim_lat_delta_low, victim_lat_delta_high, num_victims)
    lons = victim_lon + np.random.uniform(victim_lon_delta_low, victim_lon_delta_high, num_victims)
    v_type = np.full(num_victims, "piw")

    ## These will be removed, once VictimGroup is refactored.
    x = np.full(num_victims, 0.5)
    y = np.full(num_victims, 0.5)
    z = np.full(num_victims, 1)

    # Simulation also takes a 'frame_dir' argument to specify where to save data.
    s = Simulation(simulation_center_lat, simulation_center_lon, config_path, start_date, end_date)
    vics = VictimGroup(x=x, y=y, z=z, lat=lats, lon=lons, victim_type=v_type, env=s.env, config_path=config_path)

    # Add generated victims and run simulation
    s._add_victim_group(vics)
    s.Run()

if __name__ == "__main__":
    main()
