from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from matplotlib.animation import FuncAnimation
import numpy as np
from typing import Optional
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import RegularGridInterpolator

from .environment import Environment
from application.config import Config


class Simulation:

    def __init__(self, lat: float, lon: float, config_path: str, start_date:datetime, end_date:datetime):
        self.lat = lat
        self.lon = lon
        self.config_path = config_path
        self.config = Config(self.config_path)
        self.start=start_date
        self.end=end_date

        self.env = Environment(self.lat, self.lon, self.config_path, date=start_date)
        self.currents=self.env.current_data
        self.depth=self.env.depth_data
        self.wind=self.env.wind_data

        self.fig=None
        self.ax=None
        self.current_step=0
        self.simulation_steps=100

#        self.Plot()

    def Plot(self):
        if not self.fig:
            self.fig, self.ax = plt.subplots(figsize=(10,8), subplot_kw={'projection': ccrs.PlateCarree()})
            self.ax.coastlines()
            self.ax.add_feature(cfeature.LAND, facecolor='lightgray')

        uo=self.currents.uo.values
        vo=self.currents.vo.values

        lat=self.currents.latitude.values
        lon=self.currents.longitude.values
        deptho=self.depth.deptho.values

        uw = self.wind.eastward_wind.values
        vw = self.wind.northward_wind.values

        wind_lat = np.linspace(lat.min(), lat.max(), uw.shape[0])
        wind_lon = np.linspace(lon.min(), lon.max(), uw.shape[1])
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        uw_interpolator = RegularGridInterpolator((wind_lat, wind_lon), uw, bounds_error=False, fill_value=0)
        vw_interpolator = RegularGridInterpolator((wind_lat, wind_lon), vw, bounds_error=False, fill_value=0)
        uw_grid = uw_interpolator((lat_grid, lon_grid))
        vw_grid = vw_interpolator((lat_grid, lon_grid))

        depth_min = np.nanmin(deptho)
        depth_max = np.nanmax(deptho)
        depth_contour = self.ax.contourf(lon_grid, lat_grid, deptho, levels=np.linspace(depth_min, depth_max, 20), cmap='Blues', alpha=0.7)

        self.fig.colorbar(depth_contour, ax=self.ax, label='Depth (m)')

        self.ax.quiver(lon_grid, lat_grid, uo, vo, color='red', alpha=0.7, label="Current")
        self.ax.quiver(lon_grid, lat_grid, uw_grid, vw_grid, color='green', alpha=0.7, label="Wind")

        self.ax.set_title(f"Surface Currents and Wind on {str(self.env.date)} at Step 0")
        self.ax.set_xlabel("Longitude")
        self.ax.set_ylabel("Latitude")
        self.ax.set_xlim(lon.min(), lon.max())
        self.ax.set_ylim(lat.min(), lat.max())

    def Update(self, frame):
        """
        Updates the environment for the next step in the simulation and updates data for RL model.
        Also triggers visualisation updates.
        """
        self.current_step = frame

        uo = self.currents.uo.values

    def Run(self):
        self.Plot(0)
        ani = FuncAnimation(self.fig, self.Update, frames=self.simulation_steps, interval=1000)
        plt.show()
