import math
from typing import Dict,Tuple, Optional
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
        self.bounds = self._calculate_bounds()
        self.date = self._get_random_date() if not date else date
        logger.info("\033[32mEnvironment initialized.\033[0m")
        logger.debug(f"{self.config_path=} {self.center=} {self.margin=} {self.bounds=} {self.date=}")


        self.Update()
        self.wind_interpolator = self._create_interpolator(self.wind_data, "eastward_wind", "northward_wind")
        self.current_interpolator = self._create_interpolator(self.current_data, "uo", "vo")
        
    def _calculate_bounds(self) -> Tuple[float,float,float,float]:
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
        logger.debug("Environment bounds calculated successfully.")

        return (min_lat, max_lat, min_lon, max_lon)

    def _get_random_date(self) -> datetime:
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
        result = start + timedelta(seconds=random_seconds)
        logger.info("Random date generated successfully.")
        logger.debug(f"Random date: {result}")
        return result

    def _create_interpolator(self, data: dict, u_key: str, v_key: str):
        latitudes = np.array(data["latitude"])
        longitudes = np.array(data["longitude"])
        u_val = np.array(data[u_key])
        v_val = np.array(data[v_key])

        u_interp = RegularGridInterpolator((latitudes, longitudes), u_val, bounds_error=False, fill_value=None)
        v_interp = RegularGridInterpolator((latitudes, longitudes), v_val, bounds_error=False, fill_value=None)

        return lambda lat, lon: (u_interp((lat, lon)), v_interp((lat, lon)))

    def CurrentData(self):
        """
        Fetches surface current data within the environment bounds.

        :return: Surface current data.
        """
        fetcher = CurrentFetcher(self.config_path)
        logger.debug("Current data fetched successfully.")
        return fetcher.SurfaceCurrents(self.date, self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3])

    def DepthData(self):
        """
        Fetches depth data within the environment bounds.

        :return: Depth data.
        """
        fetcher = DepthFetcher(self.config_path)
        logger.debug("Depth date fetched successfully.")
        return fetcher.DepthData(self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3])

    def WindData(self):
        """
        Fetches wind data within the environment bounds.

        :return: Wind data.
        """
        fetcher = WindFetcher(self.config_path)
        logger.debug("Wind data fetched successfully.")
        return fetcher.WindData(self.date, self.bounds[0], self.bounds[1], self.bounds[2], self.bounds[3])

    def Update(self, date:Optional[datetime]=None):

        # Does this cause issues? Nothing should be calling this function except the simulation.Tick method, but this does potentially allow arbitrary dates.
        if date:
            self.date = date
        
        self.current_data = self.CurrentData()
        self.depth_data = self.DepthData()
        self.wind_data = self.WindData()
        logger.debug(f"Environment data updated for {self.date.strftime('%d%b%Y %H:%M:%S')}")

    def Query(self, lat: float, lon: float) -> Dict[str, Tuple[float, float]]:
        """
        Query wind and current data at a given latitude and longitude.

        :param lat: Latitude of the query point.
        :param lon: Longitude of the query point.
        :return: Dictionary with "net_wind" and "net_current" as keys.
        """
        lat_min, lat_max, lon_min, lon_max = self.bounds

        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            logger.warning(f"Query point ({lat}, {lon}) is out of bounds!")
            raise ValueError(f"Coordinate ({lat}, {lon}) is out of bounds. Something went wrong.")
        
        u_wind, v_wind = self.wind_interpolator(lat, lon)
        u_cur, v_cur = self.current_interpolator(lat, lon)
        return {"net_wind": (u_wind.item(), v_wind.item()), "net_current": (u_cur.item(), v_cur.item())}
        #return {"net_current": (u_cur.item(), v_cur.item())}

if __name__ == "__main__":
    lat = 30.0
    lon = -80.0
    margin = 5 # miles
    e = Environment(lat, lon, margin)
