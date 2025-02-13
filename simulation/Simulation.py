from datetime import datetime, timedelta
from typing import Optional

from .Environment import Environment
from .Visualizer import Visualizer
from .Victim import Victim
from application.config import Config
from application.logger import Logger

logger = Logger(__name__).get()


class Simulation:

    def __init__(self, lat: float, lon: float, config_path: str, start_date:datetime, end_date:datetime):
        self.lat = lat
        self.lon = lon
        self.config_path = config_path
        self.config = Config(self.config_path)
        
        self.start=start_date
        self.end=end_date
        self.time_step=timedelta(minutes=float(self.config.get_value("environment.settings.simulation_timedelta_minutes")))
        self.date=self.start

        self.env = Environment(self.lat, self.lon, self.config_path, date=start_date)
        self.currents=self.env.current_data
        self.depth=self.env.depth_data
        self.wind=self.env.wind_data

        self.vis = Visualizer(self)

        self.victims=[]

        self.current_step=0
        self.simulation_steps=self._calculate_steps()

        logger.info({"message": "\033[32mSimulation initialized\033[0m"})
        logger.debug({"event": "simulation_object_created", "data": {"Center": (lat,lon), "StartDate":self.start.isoformat(), "EndDate":self.end.isoformat(), "TimeDelta":str(self.time_step), "VictimCount":len(self.victims), "NumSteps":self.simulation_steps}})

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
        self.current_step+=1
        self.env.Update(self.date)
        for v in self.victims:
            v.Update(self.current_step)
        logger.info({"message": f"Tick at {self.date.strftime('%d%b%Y %H:%M:%S')}", "event": f"tick_{self.current_step}|{self.simulation_steps}", "data":{"date": self.date.isoformat()}})
        
    def Run(self, file: Optional[str] = None, static:bool = False):
        if static:
            self.vis.plot(0)
            self.vis.show()
        else:
            self.vis.run(file is None)
