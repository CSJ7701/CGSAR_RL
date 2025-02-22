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

        # Set up point cloud
        self.num_cloud_points = int(self.config.get_value("victims.cloud_points"))
        self.point_cloud_noise = float(self.config.get_value("victims.cloud_noise"))
        self.point_cloud_noise_radians = float(self.config.get_value("victims.cloud_angle_noise"))
        num_victims = len(self.lats)
        self.cloud_positions = np.zeros((num_victims, self.num_cloud_points, 2))

        for i in range(num_victims):
            # For each victim, initialize the cloud as the parent's position plus small random offsets
            lat_offsets = np.random.normal(0, self.point_cloud_noise, self.num_cloud_points)
            lon_offsets = np.random.normal(0, self.point_cloud_noise, self.num_cloud_points)
            self.cloud_positions[i, :, 0] = self.lats[i] + lat_offsets
            self.cloud_positions[i, :, 1] = self.lons[i] + lon_offsets

        self.logger.debug({
            "event": "VictimGroup_created",
            "data": {
                "n_victims": len(self.lats),
                "dt": self.dt,
                "pi": self.pi,
                "areas": self.areas.tolist(),
                "cloud_points": self.num_cloud_points,
                "cloud_noise": self.point_cloud_noise
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

    def _update_point_cloud(self):
        num_points = self.cloud_positions.shape[1]

        parent_velocity = np.repeat(self.velocities[:,None,:], num_points, axis=1)

        # Update probability
        p_update = float(self.config.get_value("victims.cloud_deviation_chance"))
        update_mask = np.random.rand(parent_velocity.shape[0], parent_velocity.shape[1]) < p_update  # How this work?

        # Parent's vel in polar coords
        v_base = parent_velocity[update_mask]
        vx = v_base[:,0]
        vy = v_base[:,1]
        speed = np.sqrt(vx**2 + vy**2)
        angle = np.arctan2(vx,vy) # Order?

        # Add small angle perturbation (radians)
        angle_noise = np.random.normal(0, self.point_cloud_noise_radians, angle.shape)
        new_angle = angle-angle_noise

        # Re-compute velocity using new angle
        # Keep original magnitude as close as possible
        new_vx = speed*np.cos(new_angle)
        new_vy = speed*np.sin(new_angle)

        cloud_velocities = parent_velocity.copy()
        cloud_velocities[update_mask] = np.column_stack((new_vy, new_vx))

        # Cloud positions
        cloud_lats = self.cloud_positions[:,:,0]

        d_lat = (cloud_velocities[:,:,1] * self.dt) / self.earth_rad * (180/self.pi)
        d_lon = (cloud_velocities[:,:,0] * self.dt) / (self.earth_rad*np.cos(np.radians(cloud_lats))) * (180/self.pi)
        displacement = np.stack((d_lat, d_lon), axis=2)

        self.cloud_positions += displacement
        
        self.logger.debug({
            "event": "point_cloud_update",
            "data": {
                "mean_cloud_lat": float(np.mean(self.cloud_positions[:, :, 0])),
                "mean_cloud_lon": float(np.mean(self.cloud_positions[:, :, 1]))
            }
        })
        
    def update(self):
        steps = self._simulation_steps()

        for step in range(steps):
            self.velocities = self.env.VectorizedQuery(self.lats, self.lons)["net_current"]
            self._position()
            self._update_point_cloud()
            self.logger.debug({"event": "update_step", "step": step})

    def all_points(self):
        # Parent positions: shape (num_victims, 2)
        parent_positions = np.column_stack((self.lats, self.lons))
        # Flatten cloud_positions: shape (num_victims * num_cloud_points, 2)
        cloud_positions = self.cloud_positions.reshape(-1,2)

        all_positions = np.vstack((parent_positions, cloud_positions))
        return all_positions
        
