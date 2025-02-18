from typing import Optional
import h5py
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import logging

from application.logger import Logger

plt.set_loglevel(level='warning')
pil_logger = logging.getLogger('PIL')
pil_logger.setLevel(logging.WARNING)
conv_logger = logging.getLogger('h5py')
conv_logger.setLevel(logging.WARNING)
logger = Logger(__name__).get()

class Visualizer:
    def __init__(self, hdf5_file: str):
        self.data_file = hdf5_file
        self.fig, self.ax = plt.subplots(figsize=(10,8), subplot_kw={'projection': ccrs.PlateCarree()})
        self.ax.coastlines()
        self.ax.add_feature(cfeature.LAND, facecolor='lightgray')
        self.current_step = 1
        self.wind_interpolator_u = None
        self.wind_interpolator_v = None

        with h5py.File(self.data_file, "r") as f:
            self.total_steps = len(f.keys())  # Count number of steps stored

        logger.info({
            "message": "\033[32mVisualizer initialized.\033[0m",
            "event": "visualizer_object_created",
            "data": {"total_steps": self.total_steps, "h5py_file": self.data_file}
        })

        print(self.total_steps)

    
    def _load_step_data(self, step_number):
        with h5py.File(self.data_file, "r") as file:
            step_group = file[f"step_{step_number}"]

            if not step_group:
                logger.critical({
                    'message': f"Step {step_number} does not exist.",
                    'event': "step_data_error",
                    'data': {
                        'step': step_number
                                 }
                                })
                raise ValueError(f"Step {step_number} does not exist.")

            current = step_group['current']
            depth = step_group['depth']
            wind = step_group['wind']
            self.lat = current['latitude'][:]
            self.lon = current['longitude'][:]
            self.uo = current['uo'][:]
            self.vo = current['vo'][:]
            self.deptho = depth['deptho'][:]
            
            wind_lat = wind['latitude'][:]
            wind_lon = wind['longitude'][:]
            uw = wind['eastward_wind'][:]
            vw = wind['northward_wind'][:]

            self._create_interpolators(wind_lat, wind_lon, uw, vw)

            if 'victim_positions' in step_group:
                self.victim_positions = step_group['victim_positions'][:]
            else:
                self.victim_positions = np.empty((0,2))

    def _create_interpolators(self, wind_lat, wind_lon, uw, vw):
        self.wind_interpolator_u = RegularGridInterpolator((wind_lat, wind_lon), uw, bounds_error=False, fill_value=0)
        self.wind_interpolator_v = RegularGridInterpolator((wind_lat, wind_lon), vw, bounds_error=False, fill_value=0)

    def plot(self, step):
        self._load_step_data(step)
        if not self.wind_interpolator_v or not self.wind_interpolator_u:
            logger.critical({
                'message': "Wind interpolators not initialized.",
                'event': "plot_error"
            })
            raise ValueError("Wind interpolators not initialized.")
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        uw_grid = self.wind_interpolator_u((lat_grid, lon_grid))
        vw_grid = self.wind_interpolator_v((lat_grid, lon_grid))

        # Depth countour
        depth_min, depth_max = np.nanmin(self.deptho), np.nanmax(self.deptho)
        self.depth_contour = self.ax.contourf(lon_grid, lat_grid, self.deptho, levels = np.linspace(depth_min, depth_max, 20), cmap='Blues', alpha=0.7)

        # Vectors
        self.currents = self.ax.quiver(lon_grid, lat_grid, self.uo, self.vo, color='red', alpha=0.7, label='Currents')
        self.winds = self.ax.quiver(lon_grid, lat_grid, uw_grid, vw_grid, color='green', alpha=0.7, label='Wind')

        # Victims
        self.victims = self.ax.scatter(self.victim_positions[:,1], self.victim_positions[:,0], color='purple', marker='o', label='Victims', s=10)

        self.ax.set_title(f"Surface Currents and Wind at Step {step}")
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_xlim(self.lon.min(), self.lon.max())
        self.ax.set_ylim(self.lat.min(), self.lat.max())

        #self.fig.colorbar(self.depth_contour, ax=self.ax, label='Depth (m)')

    def update(self, frame):
        self._load_step_data(frame+1)
        if not self.wind_interpolator_v or not self.wind_interpolator_u:
            logger.critical({
                'message': "Wind interpolators not initialized.",
                'event': "plot_update_error"
            })
            raise ValueError("Wind interpolators not initialized.")
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        uw_grid = self.wind_interpolator_u((lat_grid, lon_grid))
        vw_grid = self.wind_interpolator_v((lat_grid, lon_grid))
        
        self.currents.set_UVC(self.uo, self.vo)
        self.winds.set_UVC(uw_grid, vw_grid)
        self.victims.set_offsets(self.victim_positions)
        print(self.victim_positions)

        self.ax.set_title(f"Surface Currents and Wind at Step {frame+1}")
        return self.currents, self.winds, self.victims
    
    def show(self):
        plt.show()

    def run(self, show: bool = False, file: Optional[str] = None):
        self.plot(1)
        ani = anim.FuncAnimation(self.fig, self.update, frames=self.total_steps, interval = 500)

        if show:
            logger.info({
                "message": "\033[32mDisplaying plot...\033[0m",
                "event": "plot_display"
                })
            plt.show()
        else:
            save_path = file if file else "./test.mp4"
            ani.save(save_path, writer=anim.FFMpegWriter())
            logger.info({
                "message": f"Animation saved to \033[32m{save_path}\033[0m",
                "event": "plot_save",
                "data": {"file": save_path}
            })
