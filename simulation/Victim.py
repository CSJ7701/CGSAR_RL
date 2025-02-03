from application.config import Config
import logging

logger = logging.getLogger(__name__)

class Victim:
    def __init__(self, x: float, y: float, z: float, lat: float, lon: float, type:str, config_path: str):
        self.x=x
        self.y=y
        self.z=z
        self.lat=lat,
        self.lon=lon,
        self.start=(lat, lon)
        self.type=self._parse_type(type).lower()
        self.config_path=config_path
        self.config=Config(self.config_path)

        self.density = float(self.config.get_value(f"victims.{self.type}.density"))
        self.pi = float(self.config.get_value("environment.constants.pi"))
        self.volume = (4/3)*self.pi*self.x*self.y*self.z
        self.mass = self.density * self.volume

        self.position = []
        self.velocity=0

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

    def F(self, water_velocity: float):
        p = float(self.config.get_value("environment.constants.water_density"))
        A = self._csa(self.x, self.z)
        return (water_velocity**2) * A * p

    def A(self, water_velocity: float) -> float:
        F = self.F(water_velocity)
        m = self.mass
        return F/m

    def V(self, water_velocity: float) -> float:
        ...
