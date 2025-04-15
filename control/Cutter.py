import h5py
import numpy as np
from scipy.ndimage import zoom
from scipy.spatial import distance
from datetime import datetime
from application.config import Config

import matplotlib.pyplot as plt

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
        self.orientation = 0 # 0: N, 90: E, 180: S, 270: W

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

    def _rescale_heatmap_old(self, target_shape):
        scale_factors = (target_shape[0]/self.heatmap.shape[0], target_shape[1]/self.heatmap.shape[1])
        return zoom(self.heatmap, scale_factors, order=1)

    def _rescale_heatmap(self, target_shape):
        """
        Rescale the heatmap to match the target shape while preserving important features.
        """
        # Ensure heatmap not empty
        if np.all(self.heatmap == 0):
            return np.zeros(target_shape)

        # Use order=1 for bilinear interpolation, preserves gradients
        # better than nearest neighbor (order=0) while being less computationally
        # expensive than cubic (order=3)
        scale_factors = (target_shape[0]/self.heatmap.shape[0],
            target_shape[1]/self.heatmap.shape[1])

        rescaled = zoom(self.heatmap, scale_factors, order=1)

        # Preserve the sum of heatmap values to maintain probability mass
        if np.sum(self.heatmap) > 0 and np.sum(rescaled)>0:
            rescaled = rescaled * (np.sum(self.heatmap) / np.sum(rescaled))

        return rescaled

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

    def _get_local_heatmap(self, x,y,window_size):
        """
        Extract a fixed-size window from the heatmap centered on the cutter's position.
        This provides local spatial context without the dimension problems of trying to parse the whole heatmap.
        """

        grid_y = int(round(y)) + len(self.latitudes) // 2
        grid_x = int(round(x)) + len(self.longitudes) // 2

        # Ensure grid indices are within bounds
        grid_y = max(0, min(grid_y, len(self.latitudes) - 1))
        grid_x = max(0, min(grid_x, len(self.longitudes) - 1))

        #print(f"Heatmap shape: {self.rescaled_heatmap.shape}")
        #print(f"Center: ({center_x}, {center_y}), Grid pos: ({grid_x}, {grid_y})")

        # Create a padded verison of the rescaled heatmap to handle edge cases
        padded_heatmap = np.pad(self.rescaled_heatmap, window_size, mode='constant')

        padded_y = grid_y + window_size
        padded_x = grid_x + window_size

        # Extract window
        local_view = padded_heatmap[
            padded_y-window_size:padded_y+window_size+1,
            padded_x-window_size:padded_x+window_size+1
        ]

        # Normalize to 0-1 (if not there already)
        if len(local_view) == 0:
            local_view = np.zeros(padded_x, padded_y)
        max_val = np.max(local_view)
        if max_val > 0:
            local_view = local_view / max_val

        return local_view

    def _get_direction_to_heatmap(self):
        """
        Calculate a unit vector pointing from the cutter toward the highest heatmap value.
        """
        x,y = self._compute_relative_position()
        heatmap_height, heatmap_width = self.rescaled_heatmap.shape

        max_idx = np.unravel_index(np.argmax(self.rescaled_heatmap), self.rescaled_heatmap.shape)
        max_y, max_x = max_idx

        center_y = heatmap_height//2
        center_x = heatmap_width//2

        target_y = center_y-max_y
        target_x = center_x-max_x

        dx = target_x - x
        dy = target_y - y

        magnitude = np.sqrt(dx**2 + dy**2)
        if magnitude > 0:
            dx /= magnitude
            dy /= magnitude

        return dx,dy

    def _get_weighted_direction(self):
        """
        Calculate a weighted direction vector considering all non-zero heatmap values.
    
        Returns:
        Tuple (dx, dy) of normalized direction components
        """
        x, y = self._compute_relative_position()
        heatmap_height, heatmap_width = self.rescaled_heatmap.shape
        center_y = heatmap_height // 2
        center_x = heatmap_width // 2
    
        # Find all non-zero heatmap cells
        non_zero_indices = np.argwhere(self.rescaled_heatmap > 0)
    
        if len(non_zero_indices) == 0:
            return 0, 0  # No direction if no non-zero values
    
        # Initialize weighted direction
        weighted_dx = 0
        weighted_dy = 0
    
        # Process each non-zero cell
        for idx in non_zero_indices:
            cell_y, cell_x = idx
        
            # Convert to relative coordinates
            target_y = (center_y - cell_y)  # Inverted y-axis
            target_x = (cell_x - center_x)
        
            # Get cell value (weight)
            cell_value = self.rescaled_heatmap[cell_y, cell_x]
        
            # Calculate direction and distance
            dx = target_x - x
            dy = target_y - y
            distance = np.sqrt(dx**2 + dy**2)
        
            # Skip if we're at the exact location to avoid division by zero
            if distance < 0.001:
                continue
            
            # Weight by value and inverse distance
            weight = cell_value / (distance + 0.1)  # Add small constant to avoid division by zero
        
            # Accumulate weighted direction
            weighted_dx += dx * weight / distance  # Normalize to unit vector before applying weight
            weighted_dy += dy * weight / distance
    
        # Normalize final direction
        magnitude = np.sqrt(weighted_dx**2 + weighted_dy**2)
        if magnitude > 0:
            weighted_dx /= magnitude
            weighted_dy /= magnitude
    
        return weighted_dx, weighted_dy    

    def _get_heatmap_distance(self,x,y) -> float:
        
        heatmap_indices = np.argwhere(self.heatmap > 0)
        if heatmap_indices.size == 0:
            return float('inf')

        heatmap_coords = np.array([
            ((self.heatmap_longitudes[j] - self.lon_center) * 60,
             (self.heatmap_latitudes[i] - self.lat_center) * 60)
            for i,j in heatmap_indices
        ])

        min_dist = np.min(distance.cdist([(x,y)], heatmap_coords))
        return min_dist

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
            self.orientation = 0
        elif direction == 'S':
            self.lat -= displacement_nm / 60
            self.orientation = 180
        elif direction == 'E':
            self.lon += displacement_nm / (60*np.cos(np.radians(self.lat)))
            self.orientation = 90
        elif direction == 'W':
            self.lon -= displacement_nm / (60*np.cos(np.radians(self.lat)))
            self.orientation = 270

        # Update the cutter's position
        self.path[f"{self.current_step}"].append((self.lat, self.lon))

    # At a height of 10 meters above the ground, visibility should be about 356.96 Km, or 192.7 nm.
    def victim_check(self, radius_nm=5):
        """
        Check if any victims are within a specified radius from the Cutter's current position.

        :param radius_nm: The radius (in nautical miles) to check for nearby victims (default is 1).

        :return: True if there are any victims within the specified radius, False otherwise (bool).
        """
        if self.victim_index is None or self.victim_position[self.victim_index].size == 0:
            return False # No victims to check
        
        radius_deg = radius_nm / 60
        lat, lon = self.lat, self.lon
        victim_position = self.victim_position[self.victim_index]

        # Euclidean Distances
        distances = np.sqrt((victim_position[0] - lat) ** 2 + (victim_position[1] - lon) ** 2)
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
        x, y = self._compute_relative_position()
        depth_under_keel = self._get_depth() - self.draft
        victim_nearby = self.victim_check()
        
        direction_x, direction_y = self._get_direction_to_heatmap()
        weighted_dx, weighted_dy = self._get_weighted_direction()

        heatmap_height, heatmap_width = self.rescaled_heatmap.shape
        center_x = heatmap_height//2
        center_y = heatmap_width//2
        grid_x = center_x - int(round(x))
        grid_y = center_y - int(round(y))
        grid_x = max(0, min(grid_x, heatmap_width - 1))
        grid_y = max(0, min(grid_y, heatmap_height - 1))
        heatmap_value = self.rescaled_heatmap[grid_y, grid_x]

        distance = self._get_heatmap_distance(x,y)
        normalized_distance = np.tanh(distance / 20.0)

        # Normalize scalars to -1 to 1
        norm_depth = np.tanh(depth_under_keel / 20.0) # Assuming semi-typical(?) depths
        norm_time = self.current_step / self.max_steps if self.max_steps > 0 else 0

        window_size = 5
        local_heatmap = self._get_local_heatmap(x,y,window_size)

        # Observation vector
        obs = np.concatenate([
            np.array([
                norm_depth,                       # Normalized depth under keel
                float(victim_nearby),             # Binary indicator if victim is nearby (Do I need this? If true then episode ends...
                direction_x,                      # Direction to max heatmap val (x)
                direction_y,                      # Direction to max heatmap val (y)
                weighted_dx,                      # Weighted direction (x)
                weighted_dy,                      # Weighted direction (y)
                normalized_distance,              # Normalized distance to nearest non-zero value
                heatmap_value,                    # Heatmap value at point
                norm_time,                        # Normalized time step
            ], dtype=np.float32),

            # Flattened local heatmap view (fixed size regardless of total map dimensions)
           # local_heatmap.flatten()
        ])
                
        # Old approach...
        #distance_to_heatmap = self._get_heatmap_distance(x,y)
        #heatmap_value_at_point = self._get_heatmap_value(x,y)
        #return np.concatenate([
        #np.array([x, y, depth_under_keel, victim_nearby, distance_to_heatmap, heatmap_value_at_point, self.time_step, self.current_step], dtype=np.float32),
        #])

        return obs
        
        
     
   
