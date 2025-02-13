from typing import Dict, Optional, Tuple
import numpy as np
from scipy.interpolate import interp2d
from datetime import timedelta

from application.logger import Logger
from application.config import Config
from simulation.Environment import Environment



class Victim:
    def __init__(self, x: float, y: float, z: float, lat: float, lon: float, victim_type: str, env:Environment, config_path: str, ID: int):
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
        self.id = ID
        self.dt = float(self.config.get_value("environment.settings.victim_timedelta_seconds")) # time delta in seconds.

        self.logger = Logger(f"Simulation.Victim{self.id}", file_prefix=f"victim{self.id}").get()

        #self.density = float(self.config.get_value(f"victims.{self.type}.density"))
        #self.volume = (4/3)*self.pi*self.x*self.y*self.z
        #self.mass = self.density * self.volume
        self.pi = float(self.config.get_value("environment.constants.pi"))        
        self.mass = float(self.config.get_value(f"victims.{self.victim_type}.avg_mass"))

        self.drag_coeff = float(self.config.get_value(f"victims.{self.victim_type}.drag_coefficient"))

        self.path = [self.start]
        self.position = [self.lat, self.lon]
        self.velocity=np.array(self._get_vectors()["net_current"])
        self.logger.debug({"event": "victim_object_created", "data": {"id": self.id, "size":(x,y,z), "position":self.start, "velocity":str(self.velocity), "type":self.victim_type, "timedelta":self.dt, "mass":self.mass, "drag_coeff":self.drag_coeff}})
        print(f"Victim {self.id} init running.")

    def _parse_type(self, input_type:str) -> str:
        allowed_types= ["piw","piw_lj"]
        if input_type.lower() not in allowed_types:
            self.logger.critical({"message":f"\"{type}\" is not a valid victim type.", "event": "victim_type_error", "data": {"id": self.id, "type": input_type, "allowed_types": allowed_types}})
            raise ValueError("Invalid victim type. Please use a valid value.")
        else: return input_type

    def _simulation_steps(self) -> int:
        simulation_timestep = timedelta(minutes=float(self.config.get_value("environment.settings.simulation_timedelta_minutes")))
        victim_timestep = timedelta(seconds=self.dt)
        return simulation_timestep // victim_timestep

    def _csa(self, x:float, z:float) -> float:
        # Currently we assume orientation does not matter.
        # In the future, we may have to calculate x and z dynamically based on orientation
        return self.pi*x*z

    def _get_vectors(self):
        vector_dict = self.env.Query(self.lat, self.lon)
        self.logger.debug({"event": f"victim_{self.id}_vector_fetch", "data":{"wind_vector": vector_dict["net_wind"], "current_vector": vector_dict["net_current"]}})
        return vector_dict

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
        self.logger.debug({"message": f"Victim Forces - F_Drive:{F_drive}, F_Drag:{F_drag}, F_Net:{F_net}", "event": f"victim_{self.id}_force_calc", "data":{"water_density":rho_water, "victim_area":A, "F_drive":str(F_drive), "F_drag":str(F_drag), "F_net":str(F_net)}})
        return F_net
        
    def A(self, F: float) -> float:
        m = self.mass
        return F/m

    def V(self, A: float) -> np.ndarray:
        new_v = self.velocity + (A*self.dt)
        return new_v

    def X(self, V: np.ndarray):
        earth_rad = float(self.config.get_value("environment.constants.earth_radius"))
        d_lat = (V[1] * self.dt) / earth_rad*(180/np.pi)
        d_lon = (V[0] * self.dt) / (earth_rad*np.cos(np.radians(self.lat)))*(180/np.pi)
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

    def Update(self, step:int=-1):
        steps = self._simulation_steps()

        for _ in range(steps):
            v_water = np.array(self._get_vectors()["net_current"])
            v_rel = v_water-self.velocity

            F_net = self.F(v_rel)
            A=self.A(F_net)
            self.velocity=self.V(A)
            self.logger.debug({"message":f"Victim Velocity - v_water:{v_water}, v_relative:{v_rel}, v_victim:{self.velocity}", "step":step, "event": f"victim_{self.id}_velocity_update", "data":{"v_water":str(v_water), "v_rel":str(v_rel), "Force": str(F_net), "Acceleration": str(A), "Velocity": str(self.velocity)}})
            self.lat, self.lon = self.X(self.velocity)

            self.position = (self.lat, self.lon)
            self.logger.debug({"message": f"Victim position update: {self.position}", "step":step, "event":f"victim_{self.id}_position_update", "data":{"position": self.position, "displacement_from_start":self.Displacement()}})
            self.path.append(self.position)

        
        
        
        
        
        

    
