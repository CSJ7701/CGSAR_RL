import math
from typing import Tuple, Optional
import os
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import RegularGridInterpolator
import logging

from application.config import Config
from .BathymetryFetcher import BathymetryFetcher
from .CurrentFetcher import CurrentFetcher
from .DepthFetcher import DepthFetcher
from .WindFetcher import WindFetcher

logger = logging.getLogger(__name__)

class Environment:
    """
    Represents an environment with oceanic and atmospheric conditions,
    fetching data on currents, depth, and wind within a specified boundary.
    """

    def __init__(self, lat: float, lon: float, config_path: str, margin:int=0, date:Optional[datetime]=None) -> None:
        """
        Initializes the Environment object with geographic location and configuration settings.
        
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
        self.date = self.get_date() if not date else date
        logger.info("Environment initialized.")
        logger.debug(f"{self.config_path=} {self.center=} {self.margin=} {self.bounds=} {self.date=}")
        
        self.Update()

    def calculate_bounds(self) -> Tuple[float,float,float,float]:
        """
        Calculates the bounding box coordinates based on the margin and center.

        :return: Tuple[min_lat, max_lat, min_lon, lax_lon]
        """
        conversion_factor = self.config.get_value("environment.settings.degrees_per_mile")
        if not conversion_factor:
            logging.warning(f"You must define a setting for 'environment.settings.degrees_per_mile' in {self.config_path}")
            raise ValueError("Error: Invalid setting for environment.settings.degrees_per_mile\nValue does not exist.")
        try:
            conversion_factor=float(conversion_factor)
        except ValueError:
            logging.warning("The value for 'environment.settings.degrees_per_mile' must be an float value.")
            raise ValueError(f"Error: Invalid setting for environment.settings.degrees_per_mile\nValue '{conversion_factor}' is not a float.")

        lat = self.center[0]
        lon = self.center[1]

        lat_margin = self.margin / conversion_factor
        lon_margin = self.margin / (conversion_factor * math.cos(math.radians(lat)))

        min_lat = lat - lat_margin
        max_lat = lat + lat_margin
        min_lon = lon - lon_margin
        max_lon = lon + lon_margin

        return (min_lat, max_lat, min_lon, max_lon)

    def get_date(self) -> datetime:
        """
        Retrieves a random date within the configured time range.

        :return: A datetime object representing a randomly chosen timestamp.
        """
        start = self.config.get_value("application.data.time_range_start")
        start = datetime.fromisoformat(start)
        end = self.config.get_value("application.data.time_range_end")
        end = datetime.fromisoformat(end)

        delta = end-start
        random_seconds = random.randint(0,int(delta.total_seconds()))
        return start + timedelta(seconds=random_seconds)

    def CurrentData(self):
        """
        Fetches surface current data within the environment bounds.

        :return: Surface current data.
        """
        fetcher = CurrentFetcher(self.config_path)
        return fetcher.SurfaceCurrents(self.date, self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3])

    def DepthData(self):
        """
        Fetches depth data within the environment bounds.

        :return: Depth data.
        """
        fetcher = DepthFetcher(self.config_path)
        return fetcher.DepthData(self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3])

    def WindData(self):
        """
        Fetches wind data within the environment bounds.

        :return: Wind data.
        """
        fetcher = WindFetcher(self.config_path)
        return fetcher.WindData(self.date, self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3])

    def Update(self):
        self.current_data = self.CurrentData()
        self.depth_data = self.DepthData()
        self.wind_data = self.WindData()

    def Plot(self):
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

if __name__ == "__main__":
    lat = 30.0
    lon = -80.0
    margin = 5 # miles
    e = Environment(lat, lon, margin)
