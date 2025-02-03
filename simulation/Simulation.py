from datetime import datetime, timedelta
import logging

from .Environment import Environment
from .Visualizer import Visualizer
from .Victim import Victim
from application.config import Config

logger = logging.getLogger(__name__)


class Simulation:

    def __init__(self, lat: float, lon: float, config_path: str, start_date:datetime, end_date:datetime, time_step: timedelta = timedelta(minutes=30)):
        self.lat = lat
        self.lon = lon
        self.config_path = config_path
        self.config = Config(self.config_path)
        
        self.start=start_date
        self.end=end_date
        self.time_step=time_step
        self.date=self.start

        self.env = Environment(self.lat, self.lon, self.config_path, date=start_date)
        self.currents=self.env.current_data
        self.depth=self.env.depth_data
        self.wind=self.env.wind_data

        self.vis = Visualizer(self)

        self.victims=[]

        self.current_step=0
        self.simulation_steps=self._calculate_steps()

        logger.info("\033[32mSimulation initialized.\033[0m")
        logger.debug(f"{self.lat=}, {self.lon=}, self.start={self.start.strftime('%d%b%Y %H:%M:%S')}, self.end={self.end.strftime('%d%b%Y %H:%M:%S')}, self.time_step={self.time_step.days*24+self.time_step.seconds//3600}h {(self.time_step.seconds%3600)//60}m {self.time_step.seconds%60}s, {self.simulation_steps=}")

    def _calculate_steps(self) -> int:
        current_time = self.start
        steps=0
        while current_time < self.end:
            current_time += self.time_step
            steps += 1
        return steps

    def _add_victim(self, vic: Victim) -> None:
        self.victims.append(vic)

    def Tick(self):
        self.date += self.time_step
        self.env.Update(self.date)
        logger.info(f"Tick at {self.date.strftime('%d%b%Y %H:%M:%S')}")
        
    def RunSave(self):
        self.vis.run()

    def RunShow(self):
        self.vis.run(True)

    def RunSingle(self):
        self.vis.plot(0, False)
        self.vis.show()
