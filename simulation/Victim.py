from typing import Dict, Tuple
import logging
import numpy as np
from scipy.interpolate import interp2d

from application.config import Config
from simulation.Environment import Environment

logger = logging.getLogger(__name__)

class Victim:
    def __init__(self, x: float, y: float, z: float, lat: float, lon: float, victim_type: str, env:Environment, config_path: str):
        self.x=x
        self.y=y
        self.z=z
        self.lat=np.float64(lat)
        self.lon=np.float64(lon)
        self.start=(self.lat, self.lon)
        self.victim_type=self._parse_type(victim_type).lower()
        self.env=env
        self.config_path=config_path
        self.config=Config(self.config_path)
        self.dt = 1

        #self.density = float(self.config.get_value(f"victims.{self.type}.density"))
        #self.volume = (4/3)*self.pi*self.x*self.y*self.z
        #self.mass = self.density * self.volume
        self.pi = float(self.config.get_value("environment.constants.pi"))        
        self.mass = float(self.config.get_value(f"victims.{self.victim_type}.avg_mass"))

        self.drag_coeff = float(self.config.get_value(f"victims.{self.victim_type}.drag_coefficient"))

        self.path = [self.start]
        self.position = [self.lat, self.lon]
        self.velocity=np.array(self._get_vectors()["net_current"])

    def _parse_type(self, input_type:str) -> str:
        allowed_types= ["piw","piw_lj"]
        if input_type.lower() not in allowed_types:
            logger.critical(f"\"{type}\" is not a valid victim type.")
            raise ValueError("Invalid victim type. Please use a valid value.")
        else: return input_type

    def _csa(self, x:float, z:float) -> float:
        # Currently we assume orientation does not matter.
        # In the future, we may have to calculate x and z dynamically based on orientation
        return self.pi*x*z

    def _get_vectors(self):
        return self.env.Query(self.lat, self.lon)

    def F(self, v_rel) -> np.array:
        rho_water = float(self.config.get_value("environment.constants.water_density"))
        A = self._csa(self.x, self.z)
        # if np.linalg.norm(v_rel) > np.linalg.norm(v_water):
        #     v_rel=v_water

        if np.linalg.norm(v_rel) > 0:
            F_drive = rho_water * A * (np.linalg.norm(v_rel) ** 2) * (v_rel / np.linalg.norm(v_rel))
        else:
            F_drive = np.array([0.0,0.0])

        if np.linalg.norm(v_rel) > 0:
            F_drag = self.drag_coeff * rho_water * A * (np.linalg.norm(v_rel)**2) * (v_rel/np.linalg.norm(v_rel))
        else:
            F_drag = np.array([0.0,0.0])

        F_net = F_drive - F_drag
        logger.debug(f"Victim Forces - F_Drive:{F_drive}, F_Drag:{F_drag}, F_Net:{F_net}")
        return F_net
        
    def A(self, F: float) -> float:
        m = self.mass
        return F/m

    def V(self, A: float) -> np.ndarray:
        new_v = self.velocity + (A*self.dt)
        return new_v

    def X(self, V: np.ndarray):
        earth_rad = float(self.config.get_value("environment.constants.earth_radius"))
        d_lat = (V[0] * self.dt) / earth_rad*(180/np.pi)
        d_lon = (V[1] * self.dt) / (earth_rad*np.cos(np.radians(self.lat)))*(180/np.pi)
        return self.lat+d_lat, self.lon+d_lon

    def Displacement(self) -> float:
        """
        Calculate net displacement (in meters) from starting position.
        Uses Haversine Formula.
        """
        earth_rad = float(self.config.get_value("environment.constants.earth_radius"))

        lat1, lon1 = map(float, self.start)
        lat2, lon2 = map(float, self.position)

        d_lat = np.radians(lat2-lat1)
        d_lon = np.radians(lon2-lon1)

        a = np.sin(d_lat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(d_lon / 2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return earth_rad*c # Distance in meters

    def Update(self):
        v_water = np.array(self._get_vectors()["net_current"])
        v_rel = v_water-self.velocity

        F_net = self.F(v_rel)
        A=self.A(F_net)
        self.velocity=self.V(A)
        logger.debug(f"Victim Velocity - v_water:{v_water}, v_relative:{v_rel}, v_victim:{self.velocity}")
        self.lat, self.lon = self.X(self.velocity)

        self.position = (self.lat, self.lon)
        logger.debug(f"Victim position update: {self.position}")
        self.path.append(self.position)

        
        
        
        
        
        

    
