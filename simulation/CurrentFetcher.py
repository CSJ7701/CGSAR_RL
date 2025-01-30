from application.config import Config

import copernicusmarine
import os
from datetime import datetime
import xarray as xr

class CurrentFetcher:
    """
    Fetches and manages ocean current datasets from Copernicus Marine.
    """
    def __init__(self, config_path: str) -> None:
        """
        Initializes the CurrentFetcher with configuration settings.

        :param config_path: Path to the configuration file.
        """
        self.c = Config(config_path)

        data_subdir = self.c.get_value("application.data.storage")
        root_path = self.c.get_value("application.settings.project_dir")
        self.data_dir = os.path.join(root_path, data_subdir)
        self.data_file = self.c.get_value("application.data.current.file")
        self.data_expiration = self.c.get_value("application.data.expiration")
        self.data_updated = self.c.get_value("application.data.current.updated")

        self.data_full = None # Placeholder

        if not self.ValidDataset_p():
            self.FetchDataset()
        self.LoadDataset()
        
    def ValidDataset_p(self) -> bool:
        """
        Checks whether the stored dataset is valid based on expiration and existence.

        :return: True if dataset is valid, False otherwise.
        """
        path = os.path.join(self.data_dir, self.data_file)
        if not isinstance(datetime.strptime(self.data_updated, '%Y-%m-%dT%H:%M:%S'), datetime):
            return False
        if os.path.exists(path):
            last_updated = datetime.strptime(self.data_updated, '%Y-%m-%dT%H:%M:%S')
            now = datetime.now()
            diff = now - last_updated
            days = diff.days
            if days > int(self.data_expiration):
                return False
            return True
        else:
            return False

    def FetchDataset(self) -> None:
        """
        Fetches the dataset from the Copernicus Marine service and updates metadata.
        """
        path = os.path.join(self.data_dir, self.data_file)
        if os.path.exists(path):
            os.remove(path)

        dataset_start = datetime.strptime(self.c.get_value("application.data.time_range_start"), '%Y-%m-%dT%H:%M:%S').isoformat()
        dataset_end = datetime.strptime(self.c.get_value("application.data.time_range_end"), '%Y-%m-%dT%H:%M:%S').isoformat()
        
        copernicusmarine.subset(
            username = self.c.get_value("application.settings.copernicus_username"),
            password = self.c.get_value("application.settings.copernicus_password"),
            dataset_id = self.c.get_value("application.data.current.ID"),
            variables = ["uo", "vo"],
            start_datetime = dataset_start,
            end_datetime = dataset_end,
            minimum_latitude = float(self.c.get_value("environment.settings.latitude_min")),
            maximum_latitude = float(self.c.get_value("environment.settings.latitude_max")),
            minimum_longitude = float(self.c.get_value("environment.settings.longitude_min")),
            maximum_longitude = float(self.c.get_value("environment.settings.longitude_max")),
            output_filename = self.data_file,
            output_directory = self.data_dir
            )
        self.c.set_value("application.data.current.updated", datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))

    def LoadDataset(self) -> None:
        """
        Loads the dataset from the stored NetCDF file.
        """
        self.data_full = xr.open_dataset(os.path.abspath(os.path.join(self.data_dir, self.data_file)))

    def SurfaceCurrents(self, date: str, min_lat: float, max_lat: float, min_lon: float, max_lon:float):
        """
        "Retrieves surface current data for a given date and geographical bounds.

        :param date: Date string, in IDO format.
        :param min_lat: Minimum latitude.
        :param max_lat: Maximum latitude.
        :param min_lon: Minimum longitude.
        :param max_lon: Maximum longitude.
        :return: Surface current data within the specified bounds.
        :raises ValueError: If dataset is not loaded or if the requested data is unavailable.
        """
        if self.data_full is None:
            raise ValueError("Error: Dataset not loaded. Call 'LoadDataset()' first.")

        try:
            surface_data = self.data_full.sel(
                time=date,
                depth=0,
                method='nearest'
            ).sel(
                latitude=slice(min_lat, max_lat),
                longitude=slice(min_lon, max_lon)
            )

            return surface_data
        except KeyError as e:
            raise ValueError(f"Error: Could not find data for the supplied parameters: {e}")

    def CloseDataset(self) -> None:
        """
        Closes the loaded dataset to free resources.
        """
        if self.data_full is not None:
            self.data_full.close()
