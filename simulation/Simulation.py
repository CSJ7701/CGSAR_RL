from datetime import datetime, timedelta
from ssl import VerifyFlags
from typing import Optional
import h5py
import numpy as np
import xarray as xr

from .Environment import Environment
from .AnimationVisualizer import AnimationVisualizer
from .Victim import Victim
from .VictimGroup import VictimGroup
from application.config import Config
from application.logger import Logger

logger = Logger(__name__).get()


class Simulation:

    def __init__(self, lat: float, lon: float, config_path: str, start_date:datetime, end_date:datetime, frame_dir: str = "data/frames"):
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

        self.victim_group: VictimGroup

        self.current_step=0
        self.simulation_steps=self._calculate_steps()

        self.hdf5_path = frame_dir +"/"+ datetime.now().strftime("%Y%m%d_%H%M%S") + ".h5"
        self.batch_size = 100
        self.step_buffer = []

        logger.info({"message": "\033[32mSimulation initialized\033[0m"})
        logger.debug({"event": "simulation_object_created", "data": {"Center": (lat,lon), "StartDate":self.start.isoformat(), "EndDate":self.end.isoformat(), "TimeDelta":str(self.time_step), "NumSteps":self.simulation_steps}})

    def _calculate_steps(self) -> int:
        current_time = self.start
        steps=0
        while current_time < self.end:
            current_time += self.time_step
            steps += 1
        return steps

    def _heatmap(self):
        grid_lons = self.currents.longitude.values
        grid_lats = self.currents.latitude.values
        
        particle_lons = self.victim_group.lons
        particle_lats = self.victim_group.lats

        lon_bins = np.linspace(grid_lons.min(), grid_lons.max(), 5*len(grid_lons))
        lat_bins = np.linspace(grid_lats.min(), grid_lats.max(), 5*len(grid_lats))

        heatmap, x_edges, y_edges = np.histogram2d(
            particle_lons, particle_lats,
            bins=[lon_bins, lat_bins]
        )

        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()

        return heatmap.T, lon_bins, lat_bins
        
    def _add_victim_group(self, vics: VictimGroup) -> None:
        self.victim_group = vics

    def _write_frame(self, file, step_data):
        current_step = step_data.get("step_number", "unknown")
        date = step_data.get("timestamp")
        
        logger.info({
            "message": f"Writing step {current_step} to HDF5",
            "event": "write_frame",
            "data": {"step": current_step}
        })
        try: 
            step_group = file.create_group(f"step_{current_step}")
            step_group.attrs["timestamp"] = date

            for key, data in step_data.items():
                if key in ["step_number", "timestamp"]:
                    continue
            
                if isinstance(data, np.ndarray):
                    step_group.create_dataset(key, data=data, dtype='float64')
                elif isinstance(data, dict):
                    for var_name, var_data in data.items():
                        if isinstance(var_data, np.ndarray):
                            step_group.create_dataset(f"{key}/{var_name}", data=var_data, dtype='float64')
                        else:
                            raise TypeError(f"Unsupported data type {type(var_data)} for key {key}/{var_name}")
                else:
                    raise TypeError(f"Unsupported data type {type(data)} for key {key}")

        except TypeError as e:
            logger.error({
                "message": f"Data type error while writing step {current_step}: {str(e)}",
                "event": "hdf5_write_error",
                "data": {"step": current_step, "key": key, "error": str(e)}
            })
            raise

        except Exception as e:
            logger.critical({
                "message": f"Unexpected error while writing step {current_step}: {str(e)}",
                "event": "hdf5_unexpected_write_error",
                "data": {"step": current_step, "key": key, "error": str(e)}
            })
            raise

                                 
    def Tick(self):
        self.date += self.time_step
        self.current_step+=1
        self.env.Update(self.date)
        self.currents = self.env.current_data
        self.wind = self.env.wind_data

        if self.victim_group:
            self.victim_group.update()
            self.heatmap, self.heatmap_lon_bins, self.heatmap_lat_bins = self._heatmap()
                
        logger.info({"message": f"Tick at {self.date.strftime('%d%b%Y %H:%M:%S')}", "event": f"tick_{self.current_step}|{self.simulation_steps}", "data":{"date": self.date.isoformat()}})
        
    def Animate(self, file: Optional[str] = None, static:bool = False):
        if static:
            self.vis.plot(0)
            self.vis.show()
        else:
            self.vis.run(file is None)

    def Run(self):
        with h5py.File(self.hdf5_path, "w") as file:
            logger.info({"message": f"Saving frames to: {self.hdf5_path}", "event": "frame_file_init", "data":{"file":self.hdf5_path}})
            
            for _ in range(0, self.simulation_steps):
                self.Tick()

                step_data = {
                    "step_number": self.current_step,
                    "timestamp": self.date.isoformat(),
                    "victims": {
                        "victim_positions": self.victim_group.all_points(),
                        "heatmap": self.heatmap,
                        "heatmap_lon_bin": self.heatmap_lon_bins,
                        "heatmap_lat_bin": self.heatmap_lat_bins,
                    },
                    "current": {
                        "uo": self.currents.uo.values,
                        "vo": self.currents.vo.values,
                        "latitude": self.currents.latitude.values,
                        "longitude": self.currents.longitude.values,
                    },
                    "wind": {
                        "eastward_wind": self.wind.eastward_wind.values,
                        "northward_wind": self.wind.northward_wind.values,
                        "latitude": self.wind.latitude.values,
                        "longitude": self.wind.longitude.values,
                    },
                    "depth": {
                        "deptho": self.depth.deptho.values,
                        "mask": self.depth.mask.values,
                    },
                }

                self.step_buffer.append(step_data)

                if len(self.step_buffer) >= self.batch_size:
                    for step in self.step_buffer:
                        self._write_frame(file, step)
                    self.step_buffer.clear()
            for step in self.step_buffer:
                self._write_frame(file, step)
                    
