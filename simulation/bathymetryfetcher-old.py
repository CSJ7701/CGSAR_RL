from application.config import Config

import rasterio
from rasterio.windows import Window
import os
import math

class BathymetryFetcher:
    def __init__(self, filepath, config_path):
        self.file = filepath
        self.config = Config(config_path)

        if os.path.exists(self.file):
            with rasterio.open(self.file) as dataset:
                self.crs = dataset.crs
                self.transform = dataset.transform
                self.res = dataset.res
        else:
            raise(ValueError(f"ERROR: file {self.file} does not exist"))
        
    def miles_to_units(self, miles, lat=None):
        if self.crs.is_geographic:
            conversion_factor = self.config.get_value("environment.settings.degrees_per_mile")
            if conversion_factor:
                conversion_factor = float(conversion_factor)
            else:
                raise ValueError("Error: Invalid settings for environment.settings.degree_per_mile.\nValue does not exist.")
            if lat:
                degrees_per_mile = 1/(conversion_factor * math.cos(math.radians(lat)))
            else:
                degrees_per_mile = conversion_factor
            return miles * degrees_per_mile
        else:
            meters_per_mile = self.config.get_value("environment.settings.meters_per_mile")
            return miles * meters_per_mile

    def calculate_window_size(self, margin_miles):
        margin_units = self.miles_to_units(margin_miles)
        window_size_x = margin_units / float(self.res[0]) # convert to pixels (cols)
        window_size_y = margin_units / float(self.res[1]) # convert to pixels (rows)
        return int(window_size_x), int(window_size_y)

    def create_window(self, lat, lon, margin_miles):
        with rasterio.open(self.file) as dataset:
            row, col = dataset.index(lon, lat) # (lon, lat) order
            window_size_x, window_size_y = self.calculate_window_size(margin_miles)
            # Create the window
            window = Window(
                col_off = col-window_size_x,
                row_off = row-window_size_y,
                width=2 * window_size_x,
                height=2 * window_size_y,
            )
            return window

    def window(self, lat, lon, margin_miles):
        window = self.create_window(lat, lon, margin_miles)
        with rasterio.open(self.file) as dataset:
            data = dataset.read(1, window=window)
            return data
            
            
