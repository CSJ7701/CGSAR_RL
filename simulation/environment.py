import math
from typing import Tuple
import os

from application.config import Config
from .BathymetryFetcher import BathymetryFetcher

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
        self.bathymetry_data = self.BathymetryData()

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

    def BathymetryData(self):
        data_path = self.config.get_value("environment.settings.bathymetry_data_path")
        if data_path:
            filepath = self.relative_filepath(data_path)
            fetcher = BathymetryFetcher(filepath, self.config_path)
            lat = self.center[0]
            lon = self.center[1]
            return fetcher.window(lat, lon, self.margin)
        else:
            raise(ValueError(f"Error: 'environment.settings.bathymetry_data_path' is not defined in the configuration file at {self.config_path}"))

    def relative_filepath(self, path: str) -> str:
        project_dir=self.config.get_value("application.settings.project_dir")
        if project_dir:
            abs_path = os.path.join(project_dir, path)
            return os.path.abspath(abs_path)
        else:
            print("Error: Invalid settings for application.settings.project_dir.\nValue does not exist.")
            exit(1)
        
if __name__ == "__main__":
    lat = 30.0
    lon = -80.0
    margin = 5 # miles
    e = Environment(lat, lon, margin)
