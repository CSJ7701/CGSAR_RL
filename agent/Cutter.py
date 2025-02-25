import h5py
import numpy as np
from datetime import datetime
from application.config import Config

class Cutter:
    def __init__(self, hdf5_path: str, lat:float, lon: float, config_path:str, initial_step:int = 1):
        self.lat = lat
        self.lon = lon
        self.data_path = hdf5_path

        self.config=Config(config_path)

        self.speed_knots = 50
        self.draft = 15
        self.time_step = None

        self.current_step = None
        
        self.path = {"start": [(lat,lon)]}

        self._load_step_data(initial_step)

    def _load_step_data(self, step):
        step_title = f"step_{step}"
        with h5py.File(self.data_path, 'r') as data:
            self.victim_position = data[f"{step_title}/victims/victim_positions"][:]

            if self.current_step is None:
                self.latitudes = data[f"{step_title}/current/latitude"][:]
                self.longitudes = data[f"{step_title}/current/longitude"][:]
                self.depth = data[f"{step_title}/depth/deptho"][:]
                
                t1 = datetime.fromisoformat(data["step_1"].attrs["timestamp"])
                t2 = datetime.fromisoformat(data["step_2"].attrs["timestamp"])
                self.time_step = (t2-t1).total_seconds()
            
        self.current_step = step
        self.path[f"{self.current_step}"] = []

    def _get_depth(self):
        lat_idx = np.abs(self.latitudes - self.lat).argmin()
        lon_idx = np.abs(self.longitudes - self.lon).argmin()
        return self.depth[lat_idx, lon_idx]

    def is_aground(self):
        return self._get_depth() < self.draft

    def move(self, direction):
        if not self.time_step:
            raise ValueError("Time step not defined. Step data may not be loaded.")
        if not self.current_step:
            raise ValueError("Current Step unknown. Cutter object was not initialized properly")

        if self.is_aground():
            print("Boat is aground!")
            return

        # Convert knots to nautical miles per second
        speed_nm_per_sec = self.speed_knots / 3600
        # Calculate displacement
        displacement_nm = speed_nm_per_sec * self.time_step

        # Approximate conversion (1 nm ~~ 1/60th of a degree)
        if direction == 'N':
            self.lat += displacement_nm / 60
        elif direction == 'S':
            self.lat -= displacement_nm / 60
        elif direction == 'E':
            self.lon += displacement_nm / (60*np.cos(np.radians(self.lat)))
        elif direction == 'W':
            self.lon -= displacement_nm / (60*np.cos(np.radians(self.lat)))

        self.path[f"{self.current_step}"].append((self.lat, self.lon))

    def victim_check(self, radius_nm=1):
        if self.victim_position.size == 0:
            return False
        
        radius_deg = radius_nm / 60
        lat, lon = self.lat, self.lon
        victim_position = self.victim_position

        distances = np.sqrt((victim_position[:,0] - lat) ** 2 + (victim_position[:,1] - lon) ** 2)
        nearby_victims = np.any(distances < radius_deg)

        return nearby_victims
