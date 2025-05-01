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

    def _get_heatmap_value_old(self, x,y) -> int: # ISN'T WORKING. INDEX /W LAT LON, NOT X/Y
        row_idx = int(round(y))
        col_idx = int(round(x))

        if 0 <= row_idx < self.heatmap.shape[0] and 0 <= col_idx < self.heatmap.shape[1]:
            return self.heatmap[row_idx, col_idx]
        else:
            return 0

    def _get_heatmap_value(self, lat, lon) -> float:
        """
        Get the heatmap value at the specified latitude and longitude.

        :param lat: Latitude position
        :param lon: Longitude position
        :return: The heatmap value at the specified position.
        """
        # Closest indices in the heatmap grid
        lat_idx=  np.abs(self.heatmap_latitudes - lat).argmin()
        lon_idx=  np.abs(self.heatmap_longitudes - lon).argmin()

        # Chekc if indices are in bounds
        if 0 <= lat_idx < self.heatmap.shape[0] and 0 <= lon_idx < self.heatmap.shape[1]:
            return self.heatmap[lat_idx, lon_idx]
        else:
            return 0.0

    def _get_direction_to_heatmap(self):
        """
        Calculate a unit vector pointing from the cutter toward the highest heatmap value.
        """
        max_idx = np.unravel_index(np.argmax(self.heatmap), self.heatmap.shape)
        max_lat_idx, max_lon_idx = max_idx

        # Lat/Lon of max value
        max_lat  = self.heatmap_latitudes[max_lat_idx]
        max_lon = self.heatmap_longitudes[max_lon_idx]

        # Convert to nautical miles
        current_x = (self.lon - self.lon_center) * np.cos(np.radians(self.lat_center)) * 60
        current_y = (self.lat - self.lat_center) * 60

        target_x = (max_lon - self.lon_center) * np.cos(np.radians(self.lat_center)) * 60
        target_y = (max_lat - self.lat_center) * 60

        dx = target_x - current_x
        dy = target_y - current_y

        # Normalize to unit vector
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
        current_x = (self.lon - self.lon_center) * np.cos(np.radians(self.lat_center)) * 60
        current_y = (self.lat - self.lat_center) * 60
    
        # Find all non-zero heatmap cells
        non_zero_indices = np.argwhere(self.heatmap > 0)
    
        if len(non_zero_indices) == 0:
            return 0, 0  # No direction if no non-zero values
    
        # Initialize weighted direction
        weighted_dx = 0
        weighted_dy = 0
    
        # Process each non-zero cell
        for idx in non_zero_indices:
            lat_idx, lon_idx = idx

            cell_lat = self.heatmap_latitudes[lat_idx]
            cell_lon = self.heatmap_longitudes[lon_idx]
        
            # Convert to relative coordinates
            target_x = (cell_lon - self.lon_center) * np.cos(np.radians(self.lat_center)) * 60
            target_y = (cell_lat - self.lat_center) * 60
        
            # Get cell value (weight)
            cell_value = self.heatmap[lat_idx, lon_idx]
        
            # Calculate direction and distance
            dx = target_x - current_x
            dy = target_y - current_y
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

    def _get_local_heatmap(self, window_size=5):
        """
        Extract a local window of the heatmap centered at the cutter's current position,
        using latitude and longitude coordinates for indexing.

        :param window_size: Radius of the window in grid cells (default 5, produces 11x11 grid)
        :return: a normalized local heatmap window as a numpy array.
        """
        lat_idx = np.abs(self.heatmap_latitudes - self.lat).argmin()
        lon_idx = np.abs(self.heatmap_longitudes - self.lon).argmin()

        # Heatmap dimensions
        heatmap_height, heatmap_width = self.heatmap.shape

        # Empty local heatmap
        local_heatmap = np.zeros((2*window_size+1, 2*window_size+1))

        # Fill in with values from global heatmap
        for i in range(2*window_size+1):
            for j in range(2*window_size+1):
                # Calculate indices
                hm_lat_idx = lat_idx + (i - window_size)
                hm_lon_idx = lon_idx + (j - window_size)

                # Check if in bounds
                if 0 <= hm_lat_idx < heatmap_height and 0 <= hm_lon_idx < heatmap_width:
                    local_heatmap[i,j] = self.heatmap[hm_lat_idx, hm_lon_idx]

        # Normalize
        max_val = np.max(local_heatmap)
        if max_val > 0:
            local_heatmap = local_heatmap / max_val

        return local_heatmap

    def calculate_visibility_mask(self, max_range_nm=10, cone_angle=60, hi_vis_radius_nm=1):
        """
        Calculate a visibility mask based on cutter orientation and position.

        :param max_range_nm: Maximum visibility range in nautical miles
        :param cone_angle: Angle of forward visibility cone in degrees
        :param hi_vis_radius_nm: Radius of high visibility circle around the cutter in nautical miles
        :return: Visibility mask matching heatmap dimensions, values 0-1
        """

        # Create empty mask matching heatmap dimensions
        mask = np.zeros_like(self.heatmap)

        # Convert nm to degrees for calc
        lat_deg_per_nm = 1/60
        lon_deg_per_nm = 1/(60*np.cos(np.radians(self.lat)))
        # Get matrix indices
        lat_idx = np.abs(self.heatmap_latitudes - self.lat).argmin()
        lon_idx = np.abs(self.heatmap_longitudes - self.lon).argmin()
        # Range indices
        max_lat_range = int(max_range_nm * lat_deg_per_nm / (self.heatmap_latitudes[1] - self.heatmap_latitudes[0]))
        max_lon_range = int(max_range_nm * lon_deg_per_nm / (self.heatmap_longitudes[1] - self.heatmap_longitudes[0]))
        # High Vis indices
        high_vis_radius_lat = int(hi_vis_radius_nm * lat_deg_per_nm / (self.heatmap_latitudes[1] - self.heatmap_latitudes[0]))
        high_vis_radius_lon = int(hi_vis_radius_nm * lon_deg_per_nm / (self.heatmap_longitudes[1] - self.heatmap_longitudes[0]))

        # for each point in the vis range
        height, width = mask.shape
        for i in range(max(0, lat_idx - max_lat_range), min(height, lat_idx + max_lat_range)):
            for j in range(max(0, lon_idx - max_lon_range), min(width, lon_idx + max_lon_range)):
                # Calculate distance from cutter in grid units
                dy = i - lat_idx
                dx = j - lon_idx

                # Calculate distance in nm
                dist_nm = np.sqrt((dy*(self.heatmap_latitudes[1] - self.heatmap_latitudes[0]) / lat_deg_per_nm)**2 +
                    (dx * (self.heatmap_longitudes[1] - self.heatmap_longitudes[0]) / lon_deg_per_nm)**2)

                # High vis circle around cutter
                if dist_nm <= hi_vis_radius_nm:
                    mask[i,j] = 1.0
                    continue

                # Skip if beyond max range
                if dist_nm > max_range_nm:
                    continue

                angle = np.degrees(np.arctan2(dx, dy)) # Negate dy, latitude increases northward.
                angle = (angle - self.orientation) % 360

                # Check if in forward facing cone
                if angle <= cone_angle/2 or angle >= 360 - cone_angle/2:
                    # Calculate visibility decay based on distance
                    visibility = 1.0 - (dist_nm / max_range_nm)**1.5 # Non linear decay
                    mask[i,j] = max(mask[i,j], visibility)
        return mask

    def store_visibility_mask(self):
        if not hasattr(self, 'mask_history'):
            self.mask_history = {}

        self.mask_history[self.current_step] = {
            'mask': self.calculate_visibility_mask()
        }

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
            # print("Boat is aground!")
            return # No movement if the cutter is aground

        # Convert knots to nautical miles per second
        speed_nm_per_sec = self.speed_knots / 3600
        # Calculate displacement
        displacement_nm = speed_nm_per_sec * self.time_step

        turn_angles = {
            'left': -45,
            'right': 45,
            'forward': 0,
            'forward_left': -15,
            'forward_right': 15
        }

        if direction in turn_angles:
            self.orientation = (self.orientation + turn_angles[direction]) % 360
        else:
            raise ValueError(f"Unknown direction: {direction}")

        # Calculate movement vector based on orientation
        # For any pure left/right turns, reduce forward movement
        move_factor = 1.0
        if direction == 'left' or direction == 'right':
            move_factor = 0.2

        # Convert to radians
        orientation_rad = np.radians(self.orientation)

        dx = displacement_nm * move_factor * np.sin(orientation_rad)
        dy = displacement_nm * move_factor * np.cos(orientation_rad)

        # Convert displacement to lat/lon changes
        # Latitude: 1nm ~~ 1/60 degrees
        lat_change = dy/60
        # Longitude: 1nm ~~ 1/(60*cos(lat))
        lon_change = dx / (60*np.cos(np.radians(self.lat)))

        self.lat += lat_change
        self.lon += lon_change

        self.path[f"{self.current_step}"].append((self.lat, self.lon))

    # At a height of 10 meters above the ground, visibility should be about 356.96 Km, or 192.7 nm.
    def victim_check_omni(self, radius_nm=5):
        """
        Check if any victims are within a specified radius from the Cutter's current position.

        :param radius_nm: The radius (in nautical miles) to check for nearby victims (default is 1).

        :return: True if there are any victims within the specified radius, False otherwise (bool).
        """
        if self.victim_index is None:
            return False
        if self.victim_position[self.victim_index].size == 0:
            return False # No victims to check
        
        radius_deg = radius_nm / 60
        lat, lon = self.lat, self.lon
        victim_position = self.victim_position[self.victim_index]

        # Euclidean Distances
        distances = np.sqrt((victim_position[0] - lat) ** 2 + (victim_position[1] - lon) ** 2)
        nearby_victims = np.any(distances < radius_deg)
        if nearby_victims:
            print("Victim Found!")
        return nearby_victims

    def victim_check(self, radius_nm=5):
        """
        Check if any victims are detected within visibility ragne based on mask.

        :return: True if any victims are detected, False otherwise
        """
        if self.victim_index is None:
            return False
        if self.victim_position is None:
            return False
        if self.victim_position.size == 0:
            return False
        if self.victim_position[self.victim_index].size == 0:
            return False # No victim to check

        visibility_mask = self.calculate_visibility_mask()
        victim_position = self.victim_position[self.victim_index]

        # Closest grid indices for victim
        lat_idx = np.abs(self.heatmap_latitudes - victim_position[0]).argmin()
        lon_idx = np.abs(self.heatmap_longitudes - victim_position[1]).argmin()

        # Check if within mask bounds
        if 0 <= lat_idx < visibility_mask.shape[0] and 0 <= lon_idx < visibility_mask.shape[1]:
            # Get probability at victim location
            vis_probability = visibility_mask[lat_idx, lon_idx]
            detected = np.random.random() < vis_probability

            if detected:
                print("Victim Found!")
                return True
        return False

    def update(self, direction):
        """
        Update the Cutter's position and check for nearby victims, moving it to the next step.
        
        :param direction: The direction to move the Cutter ('forward', 'forward_left', 'forward_right', 'right, or 'left').
        
        :raises ValueError: If the current step is not defined.
        """
        if not self.current_step:
            raise ValueError("Current step is unknown. Cutter object was not initialized properly")

        self.move(direction)
        self.victim_check()
        self.store_visibility_mask()
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

        heatmap_value = self._get_heatmap_value(self.lat, self.lon)

        distance = self._get_heatmap_distance(x,y)
        normalized_distance = np.tanh(distance / 20.0)

        # Normalize scalars to -1 to 1
        norm_depth = np.tanh(depth_under_keel / 20.0) # Assuming semi-typical(?) depths
        norm_time = self.current_step / self.max_steps if self.max_steps > 0 else 0

        # Local heatmap
        window_size = 5
        local_heatmap = self._get_local_heatmap(window_size)

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
            local_heatmap.flatten()
        ])

        # Old approach...
        #distance_to_heatmap = self._get_heatmap_distance(x,y)
        #heatmap_value_at_point = self._get_heatmap_value(x,y)
        #return np.concatenate([
        #np.array([x, y, depth_under_keel, victim_nearby, distance_to_heatmap, heatmap_value_at_point, self.time_step, self.current_step], dtype=np.float32),
        #])

        return obs

    def observe_dict(self):
        # Get relevant data
        x, y = self._compute_relative_position()
        depth_under_keel = self._get_depth() - self.draft
    
        # Navigation data
        direction_x, direction_y = self._get_direction_to_heatmap()
        weighted_dx, weighted_dy = self._get_weighted_direction()
        distance = self._get_heatmap_distance(x, y)
        normalized_distance = np.tanh(distance / 20.0)
        #heatmap_value = self._get_heatmap_value(x, y)
    
        # Agent state data
        norm_depth = np.tanh(depth_under_keel / 20.0)
        norm_time = self.current_step / self.max_steps if self.max_steps > 0 else 0
    
        # Create local grid
        grid_size = 9
        half_size = grid_size // 2
    
        # Initialize grid with channels
        local_grid = np.zeros((grid_size, grid_size, 5), dtype=np.float32)
    
        # Get lat/lon step sizes from heatmap grid (using that as a reference)
        if len(self.heatmap_latitudes) > 1:
            lat_step = abs(self.heatmap_latitudes[1] - self.heatmap_latitudes[0])
        else:
            lat_step = 0.01 # Default if there's only one latitude value (shouldn't happen)

        if len(self.heatmap_longitudes) > 1:
            lon_step = abs(self.heatmap_longitudes[1] - self.heatmap_longitudes[0])
        else:
            lon_step = 0.01 # Default for one lon value (also shouldn't ever happen)

        visibility_mask = self.calculate_visibility_mask()

        for i in range(grid_size):
            for j in range(grid_size):
                # Calculate lat/lon location for this cell
                cell_lat = self.lat + (i-half_size) * lat_step
                cell_lon = self.lon + (j-half_size) * lon_step

                # Find nearest indices in each dataset for this lat/lon
                ocean_lat_idx = np.abs(self.latitudes - cell_lat).argmin()
                ocean_lon_idx = np.abs(self.longitudes - cell_lon).argmin()
                hm_lat_idx = np.abs(self.heatmap_latitudes - cell_lat).argmin()
                hm_lon_idx = np.abs(self.heatmap_longitudes - cell_lon).argmin()

                # Channel 0: Normalized Depth
                if (0 <= ocean_lat_idx < self.depth.shape[0] and
                    0 <= ocean_lon_idx < self.depth.shape[1]):
                    depth = self.depth[ocean_lat_idx, ocean_lon_idx]
                    local_grid[i,j,0] = np.tanh(depth/50.0)

                # Channel 1-2: Currents
                if (0 <= ocean_lat_idx < self.uo.shape[0] and
                    0 <= ocean_lon_idx < self.uo.shape[1]):
                    local_grid[i,j,1] = self.uo[ocean_lat_idx,ocean_lon_idx]
                    local_grid[i,j,2] = self.vo[ocean_lat_idx,ocean_lon_idx]

                # Channel 3: Heatmap Values
                if (0 <= hm_lat_idx < self.heatmap.shape[0] and
                    0 <= hm_lon_idx < self.heatmap.shape[1]):
                    local_grid[i,j,3] = self.heatmap[hm_lat_idx, hm_lon_idx]

                # Channel 4: Visibility Mask
                if (0 <= hm_lat_idx < visibility_mask.shape[0] and
                    0 <= hm_lon_idx < visibility_mask.shape[1]):
                    local_grid[1,j,4] = visibility_mask[hm_lat_idx, hm_lon_idx]
            
        # Return Dict observation
        return {
            'grid': local_grid,
            'agent': np.array([norm_depth, norm_time], dtype=np.float32),
            'nav': np.array([
                direction_x, direction_y,
                weighted_dx, weighted_dy,
                normalized_distance,
            ], dtype=np.float32)
        }

        
        
     
   
