import math
from typing import Tuple
import os
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import RegularGridInterpolator

from application.config import Config
from .BathymetryFetcher import BathymetryFetcher
from .CurrentFetcher import CurrentFetcher
from .DepthFetcher import DepthFetcher
from .WindFetcher import WindFetcher

class Environment:

    def __init__(self, lat: float, lon: float, config_path: str, margin:int=0):
        """
        :param lat: Starting latitude.
        :param lon: Starting longitude.
        :param config_path: Path to the configuration file. Must be an absolute path.
        :param margin: Margin, in miles, around the starting point.
        :param config_path:
        """
        self.config_path = config_path
        self.config = Config(self.config_path)
        self.center = (lat,lon)
        self.margin = int(self.config.get_value("environment.settings.default_window_margin")) if margin == 0 else margin
        self.bounds = self.calculate_bounds()
        self.date = self.get_date()
        
        self.current_data = self.CurrentData()
        self.depth_data = self.DepthData()
        self.wind_data = self.WindData()

    def get_margin(self, input) -> int:
        default = self.config.get_value("environment.settings.default_window_margin")
        if input != 0:
            return input
        elif default:
            return int(default)
        else:
            raise ValueError("Error: Invalid settings for environment.settings.default_window_margin.\nValue does not exist.")

    def calculate_bounds(self) -> Tuple[float,float,float,float]:
        """
        Calculate the bounded area using margin and center.

        :return: Tuple[min_lat, max_lat, min_lon, lax_lon]
        """

        conversion_factor = self.config.get_value("environment.settings.degrees_per_mile")
        if not conversion_factor:
            raise ValueError("Error: Invalid setting for environment.settings.degrees_per_mile\nValue does not exist.")
        try:
            conversion_factor=float(conversion_factor)
        except ValueError:
            raise ValueError(f"Error: Invalid setting for environment.settings.degrees_per_mile\nValue '{conversion_factor}' is not an integer.")

        lat = self.center[0]
        lon = self.center[1]

        lat_margin = self.margin / conversion_factor
        lon_margin = self.margin / (conversion_factor * math.cos(math.radians(lat)))

        min_lat = lat - lat_margin
        max_lat = lat + lat_margin
        min_lon = lon - lon_margin
        max_lon = lon + lon_margin

        return (min_lat, max_lat, min_lon, max_lon)

    def get_date(self):
        start = self.config.get_value("application.data.time_range_start")
        start = datetime.fromisoformat(start)
        end = self.config.get_value("application.data.time_range_end")
        end = datetime.fromisoformat(end)

        delta = end-start
        random_seconds = random.randint(0,int(delta.total_seconds()))

        return start + timedelta(seconds=random_seconds)

    def CurrentData(self):
        fetcher = CurrentFetcher(self.config_path)
        return fetcher.SurfaceCurrents(self.date, self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3])

    def DepthData(self):
        fetcher = DepthFetcher(self.config_path)
        return fetcher.DepthData(self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3])

    def WindData(self):
        fetcher = WindFetcher(self.config_path)
        return fetcher.WindData(self.date, self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3])

    def net_current(self):
        uo = self.current_data.uo.values
        vo = self.current_data.vo.values
        lat = self.current_data.latitude.values
        lon = self.current_data.longitude.values

        tau_x = self.wind_data.eastward_stress.values
        tau_y = self.wind_data.northward_stress.values
        wind_lat = self.wind_data.latitude.values
        wind_lon = self.wind_data.longitude.values

        omega = float(self.config.get_value("environment.constants.earth_rotational_vel"))
        rho = float(self.config.get_value("environment.constants.water_density"))

        # Coriolis parameter
        f = 2 * omega * np.sin(np.deg2rad(lat))
        # Edge cases: f=0 (at the equator, for instance)
        f[np.abs(f) < 1e-8] = np.sign(f[np.abs(f) < 1e-8]) * 1e-8

        tau_x_interpolator = RegularGridInterpolator((wind_lat, wind_lon), tau_x, bounds_error=False, fill_value=0)
        tau_y_interpolator = RegularGridInterpolator((wind_lat, wind_lon), tau_y, bounds_error=False, fill_value=0)

        lon_grid, lat_grid = np.meshgrid(lon, lat)
        tau_x_interp = tau_x_interpolator((lat_grid, lon_grid))
        tau_y_interp = tau_y_interpolator((lat_grid, lon_grid))

        # Wind-induces Ekman current
        u_ekman = tau_y_interp / (rho * f[:, None])
        v_ekman = -tau_x_interp / (rho * f[:, None])

        # Net current
        net_u = uo + u_ekman
        net_v = vo + v_ekman

        return {
            'un': net_u,
            'vn': net_v,
            'latitude': lat,
            'longitude': lon
        }

    def PlotSeperate(self):
        # Extract current info
        uo = self.current_data.uo.values
        vo = self.current_data.vo.values

        lat = self.current_data.latitude.values
        lon = self.current_data.longitude.values
        deptho = self.depth_data.deptho.values
        mask = self.depth_data.mask.values

        # Wind Data
        uw = self.wind_data.eastward_wind.values
        vw = self.wind_data.northward_wind.values
        wind_lat = np.linspace(lat.min(), lat.max(), uw.shape[0])
        wind_lon = np.linspace(lon.min(), lon.max(), uw.shape[1])
        lon_grid, lat_grid = np.meshgrid(lon,lat)

        uw_interpolator = RegularGridInterpolator((wind_lat, wind_lon), uw, bounds_error=False, fill_value=0)
        vw_interpolator = RegularGridInterpolator((wind_lat, wind_lon), vw, bounds_error=False, fill_value=0)
        uw_grid = uw_interpolator((lat_grid, lon_grid))
        vw_grid = vw_interpolator((lat_grid, lon_grid))
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10,8), subplot_kw={'projection':ccrs.PlateCarree()})
        ax.coastlines()
        ax.add_feature(cfeature.LAND, facecolor='lightgray')

        depth_min = np.nanmin(deptho)
        depth_max = np.nanmax(deptho)
        depth_contour = ax.contourf(lon_grid, lat_grid, deptho, levels=np.linspace(depth_min, depth_max, 20), cmap='Blues', alpha=0.7)
        fig.colorbar(depth_contour, ax=ax, label='Depth (m)')

        # Current Vectors
        ax.quiver(lon_grid, lat_grid, uo, vo, color='red', alpha=0.7, label="Current")
        # Wind Vectors
        ax.quiver(lon_grid, lat_grid, uw_grid, vw_grid, color='green', alpha=0.7, label='Wind')
        
        ax.set_title(f"Surface Currents on {str(self.date)}")
        ax.set_xlabel("Long")
        ax.set_ylabel("Lat")
        ax.set_xlim(lon.min(), lon.max())
        ax.set_ylim(lat.min(), lat.max())

        # Display
        plt.show()

    def PlotNet(self):
        net_current = self.net_current()

        net_u = net_current['un']
        net_v = net_current['vn']
        lat = net_current['latitude']
        lon = net_current['longitude']

        fig, ax = plt.subplots(figsize = (10,8), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.coastlines()
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.quiver(lon, lat, net_u, net_v, color='blue', alpha=0.7)
        plt.show()
        
if __name__ == "__main__":
    lat = 30.0
    lon = -80.0
    margin = 5 # miles
    e = Environment(lat, lon, margin)
