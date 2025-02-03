from datetime import datetime, timedelta

from .environment import Environment
from application.config import Config


class Simulation:

    def __init__(self, lat: float, lon: float, config_path: str, start_date:datetime, end_date:datetime, time_step: timedelta = timedelta(minutes=30)):
        self.lat = lat
        self.lon = lon
        self.config_path = config_path
        self.config = Config(self.config_path)
        
        self.start=start_date
        self.end=end_date
        self.time_step=time_step

        self.env = Environment(self.lat, self.lon, self.config_path, date=start_date)
        self.currents=self.env.current_data
        self.depth=self.env.depth_data
        self.wind=self.env.wind_data

        self.current_step=0
        self.simulation_steps=100

    def Tick(self):
        ...
        
    def Run(self):
        ...
