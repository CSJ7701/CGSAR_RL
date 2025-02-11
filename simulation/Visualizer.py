import matplotlib.pyplot as plt
import matplotlib.animation as anim
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from datetime import timedelta
import logging

plt.set_loglevel (level='warning')
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)
conv_logger = logging.getLogger('h5py')
conv_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

class Visualizer:
    def __init__(self, simulation):
        """
        :param simulation: An instance of the simulation class.
        """
        self.sim = simulation
        self.fig, self.ax = plt.subplots(figsize=(10,8), subplot_kw={'projection': ccrs.PlateCarree()})
        self.ax.coastlines()
        self.ax.add_feature(cfeature.LAND, facecolor='lightgray')

        logger.info("\033[32mVisualizer initialized.\033[0m")

    def plot(self, current_step):
        env=self.sim.env
        lat, lon = env.current_data.latitude.values, env.current_data.longitude.values
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        uw, vw = env.wind_data.eastward_wind.values, env.wind_data.northward_wind.values
        wind_lat = np.linspace(lat.min(), lat.max(), uw.shape[0])
        wind_lon=  np.linspace(lon.min(), lon.max(), uw.shape[1])
        uw_interpolator = RegularGridInterpolator((wind_lat, wind_lon), uw, bounds_error=False, fill_value=0)
        vw_interpolator = RegularGridInterpolator((wind_lat, wind_lon), vw, bounds_error=False, fill_value=0)
        uw_grid = uw_interpolator((lat_grid, lon_grid))
        vw_grid = vw_interpolator((lat_grid, lon_grid))

        # Depth Contour
        depth_min, depth_max = np.nanmin(env.depth_data.deptho.values), np.nanmax(env.depth_data.deptho.values)
        self.depth_contour = self.ax.contourf(lon_grid, lat_grid, env.depth_data.deptho.values, levels=np.linspace(depth_min, depth_max, 20), cmap='Blues', alpha=0.7)

        # Ocean Currents
        self.currents = self.ax.quiver(lon_grid, lat_grid, env.current_data.uo.values, env.current_data.vo.values, color='red', alpha=0.7, label='Currents')

        # Wind Vectors
        self.winds = self.ax.quiver(lon_grid, lat_grid, uw_grid, vw_grid, color='green', alpha=0.7, label='Wind')

        # Victim plotting
        self.victims = self.ax.scatter([], [], color='purple', marker='o', label='Victims', s=5)

        self.ax.set_title(f"Surface Currents and Wind on {str(env.date)} at Step {current_step}")
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_xlim(lon.min(), lon.max())
        self.ax.set_ylim(lat.min(), lat.max())

        self.fig.colorbar(self.depth_contour, ax=self.ax, label='Depth (m)')

    def update(self, frame):
        self.sim.Tick()
        env = self.sim.env

        uo, vo = env.current_data.uo.values, env.current_data.vo.values
        lat, lon = env.current_data.latitude.values, env.current_data.longitude.values
        uw, vw = env.wind_data.eastward_wind.values, env.wind_data.northward_wind.values
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Interpolate wind data
        wind_lat = np.linspace(lat.min(), lat.max(), uw.shape[0])
        wind_lon = np.linspace(lon.min(), lon.max(), uw.shape[1])
        uw_interpolator = RegularGridInterpolator((wind_lat, wind_lon), uw, bounds_error=False, fill_value=0)
        vw_interpolator = RegularGridInterpolator((wind_lat, wind_lon), vw, bounds_error=False, fill_value=0)
        uw_grid = uw_interpolator((lat_grid, lon_grid))
        vw_grid = vw_interpolator((lat_grid, lon_grid))

        # Update ocean currents
        self.currents.set_UVC(uo, vo)
        # Update winds
        self.winds.set_UVC(uw_grid, vw_grid)

        # Update victims
        victim_lats = [v.lat for v in self.sim.victims]
        victim_lons = [v.lon for v in self.sim.victims]
        self.victims.set_offsets(np.c_[victim_lons, victim_lats])

        # Update title
        self.ax.set_title(f"Surface Currents and Wind on {str(env.date)} at Step {frame}")

        return self.currents, self.winds, self.victims
                           

    def show(self):
        plt.show()

    def run(self, show:bool=False):
        self.plot(0)
        steps = self.sim.simulation_steps
        ani = anim.FuncAnimation(self.fig, self.update, frames=steps, interval=500)
        if show:
            logger.info("\033[32mDisplaying plot...\033[0m")
            plt.show()
        else:
            ani.save("./test.mp4", writer=anim.FFMpegWriter())
            logger.info("Animation saved to \033[32m./test.mp4\033[0m")


        
