from application.logger import Logger
import numpy as np
from typing import List, Optional
from datetime import timedelta

from simulation.Environment import Environment
from application.config import Config

class VictimGroup:
    def __init__(self, x: List[float], y: List[float], z: List[float], lat: List[float], lon: List[float], victim_type: List[str], env: Environment, config_path: str):
        self.xs = np.array(x, dtype=np.float64)
        self.ys = np.array(y, dtype=np.float64)
        self.zs = np.array(z, dtype=np.float64)
        self.lats = np.array(lat, dtype=np.float64)
        self.lons = np.array(lon, dtype=np.float64)
        self.victim_types = np.array([self._parse_type(vt) for vt in victim_type])

        self.env=env
        self.config_path=config_path
        self.config = Config(config_path)
        self.logger = Logger(__name__, file_prefix="victims").get()

        # Parameters/Constants
        self.dt = float(self.config.get_value("environment.settings.victim_timedelta_seconds"))
        self.pi = float(self.config.get_value("environment.constants.pi"))
        self.earth_rad = float(self.config.get_value("environment.constants.earth_radius"))
        self.water_density = float(self.config.get_value("environment.constants.water_density"))

        self.areas = self._csa(self.xs, self.ys)

        # Lookup mass and drag coefficient per victim
        self.masses = np.array([float(self.config.get_value(f"victims.{vt.lower()}.avg_mass")) for vt in self.victim_types])
        self.drag_coeffs = np.array([float(self.config.get_value(f"victims.{vt.lower()}.drag_coefficient")) for vt in self.victim_types])

        # Initialize velocity from the current water state 
        self.velocities = self.env.VectorizedQuery(self.lats, self.lons)["net_current"]

        self.logger.debug({
            "event": "VictimGroup_created",
            "data": {
                "n_victims": len(self.lats),
                "dt": self.dt,
                "pi": self.pi,
                "areas": self.areas.tolist(),
            }
        })

    def _parse_type(self, input_type: str) -> str:
        allowed_types = ["piw", "piw_lj"]
        if input_type.lower() not in allowed_types:
            self.logger.critical({
                "message": f"'{input_type}' is not a valid victim type.",
                "event": "victim_type_error",
                "data": {"type": input_type, "allowed_types": allowed_types}
            })
            raise ValueError("Invalid victim type. Please use a valid value.")
        return input_type.lower()

    def _simulation_steps(self) -> int:
        simulation_timestep = timedelta(
            minutes = float(self.config.get_value("environment.settings.simulation_timedelta_minutes"))
        )
        victim_timestep = timedelta(seconds=self.dt)
        steps = simulation_timestep // victim_timestep
        self.logger.debug({
            "event": "simulation_steps_calculated",
            "data": {"simulation_timestep_seconds": float(simulation_timestep.seconds), "victim_timestep_seconds": float(victim_timestep.seconds), "steps": steps}
        })
        return steps

    def _csa(self, x, z):
        return self.pi * x * z

    def _position(self):
        d_lat = (self.velocities[:,1] * self.dt) / self.earth_rad * (180/self.pi)
        d_lon = (self.velocities[:,0] * self.dt) / (self.earth_rad * np.cos(np.radians(self.lats))) * (180 / self.pi)

        self.lats += d_lat
        self.lons += d_lon

        self.logger.debug({
            "event": "position_updated",
            "data": {
                "mean_position": [float(np.mean(self.lats)), float(np.mean(self.lons))]
            }})

    def update(self):
        steps = self._simulation_steps()

        for step in range(steps):
            self.velocities = self.env.VectorizedQuery(self.lats, self.lons)["net_current"]

            # Probability mask and stochastic velocity
            noise_mask = np.random.rand(len(self.lats)) < 0.5 # Change into config parameter
            noise = np.random.normal(loc=0, scale=2, size=self.velocities.shape) # Change into config parameter

            # Apply noise to masked values
            self.velocities[noise_mask] += noise[noise_mask]
            
            self._position()
            self.logger.debug({"event": "update_step", "step": step})

    def all_points(self):
        # Parent positions: shape (num_victims, 2)
        parent_positions = np.column_stack((self.lats, self.lons))
        # Flatten cloud_positions: shape (num_victims * num_cloud_points, 2)
        cloud_positions = self.cloud_positions.reshape(-1,2)

        all_positions = np.vstack((parent_positions, cloud_positions))
        return all_positions
        
