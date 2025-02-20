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

    def _forces(self):
        water_currents = self.env.VectorizedQuery(self.lats, self.lons)
        v_rel = water_currents["net_current"] - self.velocities
        norm_v_rel = np.linalg.norm(v_rel, axis=1)
        unit_v_rel = np.zeros_like(v_rel)
        nonzero = norm_v_rel > 0
        unit_v_rel[nonzero] = (v_rel[nonzero].T / norm_v_rel[nonzero]).T
        norm_sq = norm_v_rel**2

        F_drive = (self.water_density * self.areas * norm_sq)[:,None] * unit_v_rel
        F_drag = (self.drag_coeffs * self.water_density * self.areas * norm_sq)[:, None] * unit_v_rel
        F_net = F_drive - F_drag

        self.logger.debug({
            "event": "forces_computed",
            "data": {
                "mean_force": F_net.mean(axis=0).tolist()
            }})
        return F_net

    def _acceleration(self, F_net):
        accelerations = F_net / self.masses[:,None]
        self.logger.debug({
            "event": "acceleration_computed",
            "data": {
                "mean_acceleration": accelerations.mean(axis=0).tolist()
            }})
        return accelerations

    def _velocity(self, accelerations):
        self.velocities += accelerations * self.dt
        self.logger.debug({
            "event": "velocity_updated",
            "data": {
                "mean_velocity": self.velocities.mean(axis=0).tolist()
            }})

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
        """
        Update each point in the point cloud as an absolute coordinate.
        Each cloud point is updated by adding the parent's displacement (computed from velocity)
        plus a small perturbation.
        """
        # Compute displacement
        d_lat = (self.velocities[:,1] * self.dt) / self.earth_rad * (180/self.pi)
        d_lon = (self.velocities[:,0] * self.dt) / (self.earth_rad * np.cos(np.radians(self.lats))) * (180 / self.pi)
        displacement = np.stack((d_lat, d_lon), axis=1) # Shape: (num_victims, 2)

        # Random perturbation for each point in the cloud
        noise = np.random.normal(0, self.point_cloud_noise, self.cloud_positions.shape)
        # Update cloud positions: new_point = old_point + parent's displacement + noise
        self.cloud_positions += displacement[:, None, :] + noise
        self.logger.debug({"event": "point_cloud_update",
                           "data": {"mean_cloud_lat": float(np.mean(self.cloud_positions[:,:,0])),
                                    "mean_cloud_lon": float(np.mean(self.cloud_positions[:,:,1]))}})

    def update(self):
        steps = self._simulation_steps()

        for step in range(steps):
            F_net = self._forces()
            accelerations = self._acceleration(F_net)
            self._velocity(accelerations)
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
        
