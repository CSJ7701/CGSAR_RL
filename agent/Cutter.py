import h5py
import numpy as np
from datetime import datetime
from application.config import Config

class Cutter:
    """
    A class to model a USCG Cutter, capable of navigating a grid of ocean data. Cutter will move based on velocity (knots) and check for nearby victims.

    Attributes:
    lat (float): Current latitude of the Cutter.
    lon (float): Current longitude of the Cutter.
    data_path (str): Path to the HDF5 file containing environmental data.
    config (Config): Configuration object for additional settings.
    speed_knots (float): Speed of the Cutter in knots (nautical miles per hour)
    draft (float): Draft of the cutter (depth of the hull below water)
    time_step (float): Time interval between steps in seconds.
    current_step (int): The current step in the environmental simulation.
    path (dict): Dictionary of the Cutter's movement history for each step.
    victim_position (np.ndarray): Positions of victims at the current step.
    depth (np.ndarray): Depth of the ocean at each point on the grid.
    latitudes (np.ndarray): Latitude values for the grid of data.
    longitudes (np.ndarray): Longitude values for the grid of data.
    """
    def __init__(self, hdf5_path: str, lat:float, lon: float, config_path:str, initial_step:int = 1):
        """
        Initialize a Cutter instance with starting position and enviornmental data.

        :param hdf5_path: Path to the HDF5 file with environmental data.
        :param lat: Initial latitude of the Cutter.
        :param lon: Initial longitude of the Cutter.
        :param config_path: Path to the configuration file.
        :param initial_step: The initial step of the configuration to load (default is 1).
        """
        self.lat = lat
        self.lon = lon
        self.data_path = hdf5_path

        # Load configuration from the config file
        self.config=Config(config_path)

        self.speed_knots = 5 # Default speed of the cutter in knots
        self.draft = 15 # Default draft in meters
        self.time_step = None # Time step, to be determined from the data file

        self.current_step = None
        
        self.path = {"start": [(lat,lon)]} # Initialize path with starting location

        self._load_step_data(initial_step) # Load data for the initial step
        self.victim_check() # Check for victims near the cutter
        self.is_aground() # Check if the cutter is aground

    def _load_step_data(self, step):
        """
        Load data for a specific simulation step from the HDF5 file.

        :param step: The simulation step for which data is to be loaded.
        """
        step_title = f"step_{step}"
        with h5py.File(self.data_path, 'r') as data:
            self.victim_position = data[f"{step_title}/victims/victim_positions"][:]
            print(f"{step_title}/victims/victim_positions")
            # If it's the first time loading data, load the necessary grid and time step
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
        """
        Get the depth of the ocean at the Cutter's current position.

        :return: The depth of the ocean at the current position (float).
        """
        lat_idx = np.abs(self.latitudes - self.lat).argmin() # Find the closest latitude index
        lon_idx = np.abs(self.longitudes - self.lon).argmin() # Find the closest longitude index
        depth = self.depth[lat_idx, lon_idx] # Depth at current position
        return depth

    def is_aground(self):
        """
        Check if the cutter is aground (depth less than draft).

        :return: True if the Cutter is aground, False otherwise (bool).
        """
        if self._get_depth() < self.draft or np.isnan(self._get_depth()):
            aground = True
        else:
            aground = False
        return aground

    def move(self, direction):
        """
        Move the Cutter in the specified direction, considering its speed and the time step.

        :param direction: Direction to move the cutter ('N', 'S', 'E', 'W')

        :raises ValueError: If time step or current step is not defined.
        """
        if not self.time_step:
            raise ValueError("Time step not defined. Step data may not be loaded.")
        if not self.current_step:
            raise ValueError("Current Step unknown. Cutter object was not initialized properly")

        if self.is_aground():
            print("Boat is aground!")
            return # No movement if the cutter is aground

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

        # Update the cutter's position
        self.path[f"{self.current_step}"].append((self.lat, self.lon))

    def victim_check(self, radius_nm=1):
        """
        Check if any victims are within a specified radius from the Cutter's current position.

        :param radius_nm: The radius (in nautical miles) to check for nearby victims (default is 1).

        :return: True if there are any victims within the specified radius, False otherwise (bool).
        """
        if self.victim_position.size == 0:
            return False # No victims to check
        
        radius_deg = radius_nm / 60
        lat, lon = self.lat, self.lon
        victim_position = self.victim_position

        # Euclidean Distances
        distances = np.sqrt((victim_position[:,0] - lat) ** 2 + (victim_position[:,1] - lon) ** 2)
        nearby_victims = np.any(distances < radius_deg)

        return nearby_victims

    def update(self, direction):
        """
        Update the Cutter's position and check for nearby victims, moving it to the next step.
        
        :param direction: The direction to move the Cutter ('N', 'S', 'E', or 'W').
        
        :raises ValueError: If the current step is not defined.
        """
        if not self.current_step:
            raise ValueError("Current step is unknown. Cutter object was not initialized properly")
        self.move(direction)
        self.victim_check()
        self._load_step_data(self.current_step+1)

    def observe(self):
        """
        Observe the Cutter's current state, returning relevant information in the form of a NumPy array.
        
        :return: A structured NumPy array containing the Cutter's current position,
                 depth under keel, victim proximity, time step, and simulation step (np.ndarray).
        """
        lat, lon = self.lat, self.lon
        depth_under_keel = self._get_depth() - self.draft
        victim_nearby = self.victim_check()

        return np.array([lat, lon, depth_under_keel, victim_nearby, self.time_step, self.current_step], dtype=np.float32)
        
        
