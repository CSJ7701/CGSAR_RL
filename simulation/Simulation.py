import imageio
from tqdm import tqdm
from datetime import datetime, timedelta

from .environment import Environment
from application.config import Config
class Simulation:

    def __init__(self, lat: float, lon: float, config_path: str, start_time: datetime, end_time: datetime, timestep: timedelta):
        self.lat = lat
        self.lon = lon
        self.config_path = config_path
        self.start_time = start_time
        self.end_time = end_time
        self.time_step = timestep
        self.config = Config(self.config_path)
        self.frames = []

