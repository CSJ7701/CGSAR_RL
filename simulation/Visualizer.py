import matplotlib.pyplot as plt
import matplotlib.animation as anim
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from datetime import timedelta

class Visualizer:
    def __init__(self, simulation):
        """
        :param simulation: An instance of the simulation class.
        """

        self.sim = simulation
        self.fig, self.ax = plt.subplots(figsize=(10,8), subplot_kw={'projection': ccrs.PlateCarree()})
        self.ax.coastlines()
        self.ax.add_feature(cfeature.LAND, facecolor='lightgray')

    def plot(self, current_step, initial:bool):
        """
        Plots the current state of the simulation.
        """
        self.ax.clear()
        self.ax.coastlines()
        self.ax.add_feature(cfeature.LAND, facecolor='lightgray')

        env = self.sim.env
        uo, vo = env.current_data.uo.values, env.current_data.vo.values
        lat, lon = env.current_data.latitude.values, env.current_data.longitude.values
        deptho = env.depth_data.deptho.values
        uw, vw = env.wind_data.eastward_wind.values, env.wind_data.northward_wind.values

        wind_lat = np.linspace(lat.min(), lat.max(), uw.shape[0])
        wind_lon = np.linspace(lon.min(), lon.max(), uw.shape[1])
        lon_grid, lat_grid = np.meshgrid(lon,lat)
        uw_interpolator = RegularGridInterpolator((wind_lat, wind_lon), uw, bounds_error=False, fill_value=0)
        vw_interpolator = RegularGridInterpolator((wind_lat, wind_lon), vw, bounds_error=False, fill_value=0)
        uw_grid=uw_interpolator((lat_grid, lon_grid))
        vw_grid=vw_interpolator((lat_grid, lon_grid))

        depth_min, depth_max = np.nanmin(deptho), np.nanmax(deptho)
        depth_contour = self.ax.contourf(lon_grid, lat_grid, deptho, levels=np.linspace(depth_min, depth_max, 20), cmap='Blues', alpha=0.7)

        self.ax.quiver(lon_grid, lat_grid, uo, vo, color='red', alpha=0.7, label='Current')
        self.ax.quiver(lon_grid, lat_grid, uw_grid, vw_grid, color='green', alpha=0.7, label='Wind')

        self.ax.set_title(f"Surface Currents and Wind on {str(env.date)} at Step {current_step}")
        self.ax.set_xlabel('Longitude')
        self.ax.set_ylabel('Latitude')
        self.ax.set_xlim(lon.min(), lon.max())
        self.ax.set_ylim(lat.min(), lat.max())

        if initial == True:
            self.fig.colorbar(depth_contour, ax=self.ax, label='Depth (m)')

    def update(self, frame):
        self.sim.env.date += timedelta(hours=1)
        self.sim.env.Update()
        self.plot(frame, False)

    def run(self, steps=500, interval=30):
        self.plot(0, True)
        ani = anim.FuncAnimation(self.fig, self.update, frames=steps, interval=interval)
        #plt.show()
        ani.save("./test.mp4", writer=anim.FFMpegWriter(fps=60))
        
