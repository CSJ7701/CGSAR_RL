import os
import sys
import numpy as np
import random
from datetime import datetime, timedelta
import argparse
from tqdm import tqdm

from application.logger import Logger
import logging
from simulation.VictimGroup import VictimGroup
from simulation.Simulation import Simulation

# Static Variables
logger = Logger(__name__).get()
logging.basicConfig(level=logging.WARN)
proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(proj_dir,"resources/settings.json")

def parse_args():
    parser = argparse.ArgumentParser(description="Generate multiple simulation environments with varying victim sets.")

    # General Settings (Not Randomized)
    general = parser.add_argument_group("General Settings", "Static parameters for overall configuration.")
    general.add_argument("-E", "--env", type=int, required=True, help="Number of environments to generate")
    general.add_argument("-V", "--vic", type=int, required=True, help="Number of victim groups per environment")
    general.add_argument("-C", "--count", type=int, required=True, help="Number of iterations for each victim_group")
    general.add_argument("-o", "--output_dir", type=str, default=os.path.join(proj_dir, "data/simulations"), help="Directory to store simulation outputs")

    # Simulation Settings
    simulation = parser.add_argument_group("Simulation Settings", "Parameters defining the simulation environment.")
    sim_lat_group = simulation.add_mutually_exclusive_group()
    sim_lat_group.add_argument("--sim_lat", type=float, help="Base latitude for simulation")
    sim_lat_group.add_argument("--sim_lat_range", type=float, nargs=2, metavar=("LOWER", "UPPER"), help="Range for randomizing base simulation latitude.")
    sim_lon_group = simulation.add_mutually_exclusive_group()
    sim_lon_group.add_argument("--sim_lon", type=float, help="Base longitude for simulation")
    sim_lon_group.add_argument("--sim_lon_range", type=float, nargs=2, metavar=("LOWER", "UPPER"), help="Range for randomizing base simulation longitude.")
    start_group = simulation.add_mutually_exclusive_group()
    start_group.add_argument("-s", "--start", type=str, help="Simulation start date (YYYY-MM-DDTHH:MM:SS).")
    start_group.add_argument("--start_range", type=str, nargs=2, metavar=("START", "END"), help="Range for randomizing simulation start date.")
    simulation.add_argument("-d", "--duration", type=int, required=True, help="Simulation duration in hours.")

    # Victim Group Settings
    victims = parser.add_argument_group("Victim Group Settings.", "Parameters for victim group configuration.")
    victims.add_argument("-v", "--num_victims", required=True, type=int, help="Number of victims per victim group.")
    #victims.add_argument("-r", "--victim_radius", required=True, type=float, help="Radius for victim group spread. Recommended between 0.1 and 0.3.")
    vic_lat_group = victims.add_mutually_exclusive_group()
    vic_lat_group.add_argument("--vic_lat", type=float, help="Base latitude for victim group.")
    vic_lat_group.add_argument("--vic_lat_range", type=float, nargs=2, metavar=("LOWER", "UPPER"), help="Range for randomizing victim group latitude.")
    vic_lon_group = victims.add_mutually_exclusive_group()
    vic_lon_group.add_argument("--vic_lon", type=float, help="Base longitude for victim group.")
    vic_lon_group.add_argument("--vic_lon_range", type=float, nargs=2, metavar=("LOWER", "UPPER"), help="Range for randomizing victim group longitude.")

    victim_dist_group = victims.add_mutually_exclusive_group()
    victim_dist_group.add_argument("-g", "--gaussian", type=float, nargs=2, metavar=("LAT_STD_DEV", "LON_STD_DEV"), help="Use Gaussian distribution for victim positions with specified params.")
    victim_dist_group.add_argument("-u", "--uniform", type=float, nargs=1, metavar=("RADIUS"), help="Use uniform distribution for victim position with specified radius.")

    return parser.parse_args()

def generate_dateseed():
    return datetime.now().strftime("%b%d%H%M%S")

def randomize_value(value, value_range):
    if value is not None:
        return value
    if value_range:
        return random.uniform(*value_range)
    return None

def randomize_date(base_date, range_hours):
    if range_hours:
        offset = random.randint(*range_hours)
        return base_date + timedelta(hours=offset)
    return base_date

def victim_pos_gaussian(base_lat, base_lon, lat_dev, lon_dev, num = 100):
    lats = base_lat + np.random.normal(0, lat_dev, num)
    lons = base_lon + np.random.normal(0, lon_dev, num)
    return lats, lons

def victim_pos_uniform(base_lat, base_lon, r, num):
    r = np.random.uniform(0,r,num)
    theta = np.random.uniform(0, 2*np.pi, num)
    lats = base_lat + r*np.cos(theta)
    lons = base_lon + r*np.sin(theta)
    return lats, lons
    
   

def main():
    args = parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Create output directory
    dateseed = generate_dateseed() # Create a dateseed under the output dir (assuming multiple runs in output directory)
    base_output_dir = str(os.path.join(args.output_dir, dateseed))
    os.makedirs(base_output_dir, exist_ok=True)
    
    logger.info({"message": f"Saving simulations to base directory {base_output_dir}", "event": "begin_simulation_loop"})

    # === Begin ENV Loop ===
    total_iterations = args.env * args.vic * args.count
    with tqdm(total=total_iterations, desc="Total Progress") as main_bar:
        for i in tqdm(range(args.env), desc="Environments", leave=False): # Generate 'E' environments
            env_dir = os.path.join(base_output_dir, f"env_{i+1}")
            logger.info({"message": f"Initializing environment {i+1}", "event": "simulation_loop_environment", "data":{"env_dir": env_dir}})
            os.makedirs(env_dir, exist_ok=False)

            # Randomize simulation settings for each environment iteration
            sim_lat = randomize_value(args.sim_lat, args.sim_lat_range)
            sim_lon = randomize_value(args.sim_lon, args.sim_lon_range)
            if args.start:
                sim_start = datetime.strptime(args.start, "%Y-%m-%dT%H:%M:%S")
            elif args.start_range:
                start_range_dates = [datetime.strptime(date, "%Y-%m-%dT%H:%M:%S") for date in args.start_range]
                dur_days = (start_range_dates[1] - start_range_dates[0]).days
                dur_seconds = (start_range_dates[1] - start_range_dates[0]).seconds
                total_dur_hours = ((dur_days * 24) + (dur_seconds // 3600))
                range_hours = (0, total_dur_hours)
                sim_start = randomize_date(start_range_dates[0], range_hours)
            else:
                raise ValueError("Either --start or --start_range must be provided.")
            sim_end = sim_start + timedelta(hours=args.duration)


            # Create the environment
            if sim_lat and sim_lon:
                sim = Simulation(float(sim_lat), float(sim_lon), config_path, sim_start, sim_end)
                sim_min_lat, sim_max_lat, sim_min_lon, sim_max_lon = sim.env.bounds
            else:
                raise ValueError("Either sim_lat or sim_lon are not properly defined. Check your input for --lat or --lon.")

            # Print simulation settings
            # print(f"Randomized simulation settings: sim_lat={sim_lat}, sim_lon={sim_lon}, sim_start={sim_start}, sim_end={sim_end}")


        # === Begin VIC Loop ===
            for j in tqdm(range(args.vic), desc=f"Victim Groups in Env {i+1}", leave=False): # Run 'V' victim groups per environment
                vic_dir = os.path.join(env_dir, f"vics_{j+1}")
                logger.info({"message": f"Initializing victim group {j+1}", "event": "simulation_loop_victim", "data":{"vic_dir": vic_dir}})
                os.makedirs(vic_dir, exist_ok=False)

                # Randomize victim group settings for each victim iteration
                vic_lat = randomize_value(args.vic_lat, args.vic_lat_range)
                vic_lon = randomize_value(args.vic_lon, args.vic_lon_range)
                if vic_lat is None:
                    vic_lat = randomize_value(None, (sim_min_lat, sim_max_lat))
                if vic_lon is None:
                    vic_lon = randomize_value(None, (sim_min_lon, sim_max_lon))
                base_vic_lat = vic_lat if vic_lat is not None else sim_lat
                base_vic_lon = vic_lon if vic_lon is not None else sim_lon
                #victim_radius = args.victim_radius
                num_victims = args.num_victims

                # Define a buffer so victims aren't too close to the simulation boundary
                buffer_lat = (sim_max_lat - sim_min_lat) * 0.15
                buffer_lon = (sim_max_lon - sim_min_lon) * 0.15
                inset_min_lat = sim_min_lat + buffer_lat
                inset_max_lat = sim_max_lat - buffer_lat
                inset_min_lon = sim_min_lon + buffer_lon
                inset_max_lon = sim_max_lon - buffer_lon

                #r = np.random.uniform(0, victim_radius, num_victims)
                #theta = np.random.uniform(0, 2*np.pi, num_victims)
                #victim_latitudes = base_vic_lat + r*np.cos(theta)
                #victim_longitudes = base_vic_lon + r*np.sin(theta)
                #victim_latitudes = np.clip(base_vic_lat + r*np.cos(theta), inset_min_lat, inset_max_lat)
                #victim_longitudes = np.clip(base_vic_lon + r*np.sin(theta), inset_min_lon, inset_max_lon)

                if args.gaussian:
                    lat_dev, lon_dev = args.gaussian
                    victim_latitudes, victim_longitudes = victim_pos_gaussian(
                        base_vic_lat,
                        base_vic_lon,
                        lat_dev,
                        lon_dev,
                        num_victims)
                elif args.uniform:
                    radius = args.uniform[0]
                    victim_latitudes, victim_longitudes = victim_pos_uniform(
                        base_vic_lat,
                        base_vic_lon,
                        radius,
                        num_victims)
                else:
                    raise ValueError("No specified distribution...")

                victim_latitudes = np.clip(victim_latitudes, inset_min_lat, inset_max_lat)
                victim_longitudes = np.clip(victim_longitudes, inset_min_lon, inset_max_lon)
                    
                victim_types = np.full(num_victims, "piw")
                

                # Print victim group settings
                # print(f"Randomized victim group settings: base_vic_lat={base_vic_lat}, base_vic_lon={base_vic_lon}, victim_radius={victim_radius}, num_victims={num_victims}")

                # Run with output directory
                for c in tqdm(range(args.count), desc=f"Iterations in Vic Group {j+1}", leave=False):
                    victim_group = VictimGroup(lat=victim_latitudes, lon=victim_longitudes, victim_type=victim_types, env=sim.env, config_path=config_path)
                    sim.Reset()
                    sim._add_victim_group(victim_group)
                    sim.Run(save_dir=vic_dir)

                    main_bar.update(1)

if __name__ == "__main__":
    main()
