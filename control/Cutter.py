import h5py
import numpy as np
from scipy.ndimage import zoom
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

        self.speed_knots = 6 # Default speed of the cutter in knots
        self.draft = 15 # Default draft in meters
        self.time_step = None # Time step, to be determined from the data file
        self.victim_index = 0

        # Initialize current_step to None. This will be set when you call _load_step_data(), and initialized to whatever your initial step is. Then incremented on every update.
        self.current_step = initial_step
        
        self.path = {"start": [(lat,lon)]} # Initialize path with starting location

        self._load_step_data(initial_step) # Load data for the initial step
        self.victim_check() # Check for victims near the cutter
        self.is_aground() # Check if the cutter is aground

    def _load_step_data(self, step: int):
        """
        Load data for a specific simulation step from the HDF5 file.

        :param step: The simulation step for which data is to be loaded.
        """
        step_title = f"step_{step}"
        with h5py.File(self.data_path, 'r') as data:
            self.victim_position = data[f"{step_title}/victims/victim_positions"][:]
            self.heatmap = data[f"{step_title}/victims/heatmap"][:]
            self.uo = data[f"{step_title}/current/uo"][:]
            self.vo = data[f"{step_title}/current/vo"][:]
            # If it's the first time loading data, load the necessary grid and time step
            if self.time_step is None:
                self.max_steps = len(data.keys())
                self.latitudes = data[f"{step_title}/current/latitude"][:]
                self.lat_center = self.latitudes.mean()
                self.longitudes = data[f"{step_title}/current/longitude"][:]
                self.lon_center = self.longitudes.mean()
                
                self.heatmap_latitudes = data[f"{step_title}/victims/heatmap_lat_bin"][:]
                self.heatmap_longitudes = data[f"{step_title}/victims/heatmap_lon_bin"][:]
                self.depth = data[f"{step_title}/depth/deptho"][:]
                
                t1 = datetime.fromisoformat(data["step_1"].attrs["timestamp"])
                t2 = datetime.fromisoformat(data["step_2"].attrs["timestamp"])
                self.time_step = (t2-t1).total_seconds()

        self.rescaled_heatmap = self._rescale_heatmap((len(self.latitudes), len(self.longitudes)))
        self.current_step = step
        self.path[f"{self.current_step}"] = []

    def _load_true_victim(self, victim_index: int):
        """
        Set the index at which to find the 'true' victim. This determines where the real person is.

        :param victim_index: Index of the victim. Must be between 0 and the length of the victims array.
        """
        self.victim_index = victim_index

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

    def _rescale_heatmap(self, target_shape):
        scale_factors = (target_shape[0]/self.heatmap.shape[0], target_shape[1]/self.heatmap.shape[1])
        return zoom(self.heatmap, scale_factors, order=1)

    def _compute_relative_position(self):
        x = (self.lon-self.lon_center) * np.cos(np.radians(self.lat_center)) * 60 # Convert degrees to nautical miles
        y = (self.lat - self.lat_center) * 60 # Convert degrees to nautical miles
        return x,y

    def _get_heatmap_value(self, x,y) -> int:
        row_idx = int(round(y))
        col_idx = int(round(x))

        if 0 <= row_idx < self.rescaled_heatmap.shape[0] and 0 <= col_idx < self.rescaled_heatmap.shape[1]:
            return self.rescaled_heatmap[row_idx, col_idx]
        else:
            return 0

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

    def victim_check(self, radius_nm=10):
        """
        Check if any victims are within a specified radius from the Cutter's current position.

        :param radius_nm: The radius (in nautical miles) to check for nearby victims (default is 1).

        :return: True if there are any victims within the specified radius, False otherwise (bool).
        """
        if not self.victim_index or self.victim_position[self.victim_index].size == 0:
            return False # No victims to check
        
        radius_deg = radius_nm / 60
        lat, lon = self.lat, self.lon
        victim_position = self.victim_position[self.victim_index]

        # Euclidean Distances
        distances = np.sqrt((victim_position[0] - lat) ** 2 + (victim_position[1] - lon) ** 2)
        nearby_victims = np.any(distances < radius_deg)
        if nearby_victims:
            print(f"Distance: {distances}, At: {victim_position}")
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
        x, y = self._compute_relative_position()
        depth_under_keel = self._get_depth() - self.draft
        victim_nearby = self.victim_check()

        heatmap_flat = self.rescaled_heatmap.flatten()

        return np.concatenate([
            np.array([x, y, depth_under_keel, victim_nearby, self.time_step, self.current_step], dtype=np.float32),
            heatmap_flat
        ])
        
        
