from application.config import Config

class Victim:
    def __init__(self, x: float, y: float, z: float, lat: float, lon: float, config_path: str):
        self.x=x
        self.y=y
        self.z=z
        self.lat=lat,
        self.lon=lon,
        self.start=(lat, lon)
        self.config_path=config_path
        self.config=Config(self.config_path)
        self.pi = float(self.config.get_value("environment.constants.pi"))
        self.area = (4/3)*self.pi*self.x*self.y*self.z

    def _csa(self, x:float, z:float) -> float:
        # Currently we assume orientation does not matter.
        # In the future, we may have to calculate x and z dynamically based on orientation
        return self.pi*x*z

    def Force(self, water_velocity: float):
        p = float(self.config.get_value("environment.constants.water_density"))
        A = self._csa(self.x, self.z)
        return (water_velocity**2) * A * p
