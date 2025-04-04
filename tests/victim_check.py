import h5py
import numpy as np
from datetime import datetime, timedelta
import os
import matplotlib.pyplot as plt
from control.Cutter import Cutter

# Create a minimal HDF5 file for testing
test_file = "test_env.h5"
with h5py.File(test_file, "w") as f:
    for step in [1,2]:
        grp = f.create_group(f"step_{step}")

        # Create timestamp attribute (step_2 is 1 minute later)
        timestamp = datetime.now() + timedelta(minutes=(step-1))
        grp.attrs["timestamp"] = timestamp.isoformat()

        # Victims group
        vic_grp = grp.create_group("victims")
        # Create victim_positions as a dataset
        # For step 1, place one victim very close to cutter's starting point.
        # Format: [latitude, longitude]
        if step == 1:
            vic_data = np.array([[30.2005, -80.0995]])
        else:
            vic_data = np.array([[30.2005, -80.0995]])
        vic_grp.create_dataset("victim_positions", data=vic_data)

        # Create a dummy heatmap (non-zero where victim is located)
        heatmap = np.zeros((10,10))
        heatmap[5,5] = 1
        vic_grp.create_dataset("heatmap", data=heatmap)
        # Create heatmap bins arrays
        vic_grp.create_dataset("heatmap_lat_bin", data=np.linspace(30.0, 30.5, 10))
        vic_grp.create_dataset("heatmap_lon_bin", data=np.linspace(-80.5, -80.0, 10))
        
        # Current group
        cur_grp = grp.create_group("current")
        # Create dummy current datasets
        cur_grp.create_dataset("uo", data=np.zeros((10, 10)))
        cur_grp.create_dataset("vo", data=np.zeros((10, 10)))
        # Create latitude and longitude grids for the current data
        cur_grp.create_dataset("latitude", data=np.linspace(30.0, 30.5, 10))
        cur_grp.create_dataset("longitude", data=np.linspace(-80.5, -80.0, 10))
        
        # Depth group
        dep_grp = grp.create_group("depth")
        # Create a depth grid where all values exceed the draft (15 meters) except a controlled spot
        # Here we simply set the entire grid to 100 meters
        dep_grp.create_dataset("deptho", data=100 * np.ones((10, 10)))

# Now run a controlled test
# Place the cutter close to the victim: starting at (30.2000, -80.1000)
# Given the victim at (30.2005, -80.0995), this should be within a 50 nm radius (when converting to degrees, ~0.83 deg)
c = Cutter(test_file, 30.2000, -80.0000, "resources/settings.json", initial_step=1)
# Our 'dataset' has one victim.
c._load_true_victim(0)

# Check if victim is detected
if c.victim_check(radius_nm=50):
    print("Victim Detected!!")
else:
    print("Nope...")
    c.move("N")

class Visualizer:
    def __init__(self, hdf5_file: str):
        self.data_file = hdf5_file
        # Load basic grid and victim info from step_1 for plotting bounds.
        with h5py.File(self.data_file, "r") as f:
            step1 = f["step_1"]
            current = step1["current"]
            self.lats = current["latitude"][:]
            self.lons = current["longitude"][:]
            self.victim_positions = step1["victims"]["victim_positions"][:]
        self.trackline = None
        self.cutter_position = None
        self.visibility = None

    def _load_trackline(self, trackline: dict):
        self.trackline = trackline
        # Set cutter_position to the last recorded position.
        last_step = max([k for k in trackline.keys() if k != "start"], default="start")
        if last_step == "start":
            self.cutter_position = trackline["start"][-1]
        else:
            self.cutter_position = trackline[last_step][:]

    def set_visibility_radius(self, radius_nm: float):
        """Store the cutter's visibility radius in nautical miles."""
        self.visibility_radius_nm = radius_nm            

    def plot(self):
        """Plot victim positions and cutter trackline on a 2D map."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("Cutter Trackline and Victims")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        # Plot victim positions (if any)
        if self.victim_positions.size > 0:
            ax.scatter(
                self.victim_positions[:, 1],
                self.victim_positions[:, 0],
                color="red",
                marker="o",
                label="Victims"
            )

        # Plot cutter trackline if available
        if self.trackline:
            # Plot starting point
            if "start" in self.trackline:
                start_coords = self.trackline["start"]
                start_lats, start_lons = zip(*start_coords)
                ax.scatter(start_lons, start_lats, color="green", marker="x", s=100, label="Start")
            # Plot steps
            for key in sorted(self.trackline.keys()):
                if key == "start":
                    continue
                coords = self.trackline[key]
                if coords:
                    lats, lons = zip(*coords)
                    ax.plot(lons, lats, marker="o", linestyle="-", label=f"Step {key}")

        # Plot visibility radius if cutter position and radius are available
        if self.cutter_position and self.visibility_radius_nm:
            # Conversion: 1 nm is roughly 1/60 degree
            radius_deg = self.visibility_radius_nm / 60.0
            cutter_lat, cutter_lon = self.cutter_position
            circle = Circle(
                (cutter_lon, cutter_lat),
                radius_deg,
                color="blue",
                fill=False,
                linestyle="--",
                label=f"Visibility Radius ({self.visibility_radius_nm} nm)"
            )
            ax.add_patch(circle)        

        # Set axis limits based on grid from step_1
        ax.set_xlim(self.lons.min(), self.lons.max())
        ax.set_ylim(self.lats.min(), self.lats.max())
        ax.legend()
        plt.show()

v=Visualizer(test_file)
v._load_trackline(c.path)
v.set_visibility_radius(50)
v.plot()


# Clean up
os.remove(test_file)
