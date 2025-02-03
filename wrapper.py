import os
from datetime import datetime
import logging

from simulation.Simulation import Simulation
from simulation.Visualizer import Visualizer
from simulation.environment import Environment

# Static variables. These should not change.
logger = logging.getLogger(__name__)
logging.basicConfig(format='\033[91m\033[1m[%(levelname)s]\033[0m \033[33m%(name)s\033[0m %(asctime)s - %(message)s', level=logging.DEBUG)
proj_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(proj_dir,"resources/settings.json")

# Tuning parameters.
# These can change, and will likely be determined by some 'randomizer' later.
lat = 30.0
lon = -80.0
margin = 5 # miles
start_date = datetime(2023, 1, 1, 00, 00, 00)
end_date = datetime(2023, 1, 5, 00,00,00)

logger.info("Starting application...")
logger.info("Parameters are predefined for stability; this is not a final product.")
logger.info("Simulation Parameters: Center=(%s, %s), Margin=%s miles, Start=%s, End=%s",
            lat, lon, margin, start_date.strftime('%d%b%Y'), end_date.strftime('%d%b%Y')
        )

logger.info("Initializing environment.")
e = Environment(lat, lon, config_path)
logger.info("Initializing simulation.")
s = Simulation(lat, lon, config_path, start_date, end_date)
logger.info("Initializing visualizer")
v = Visualizer(s)
logger.info("Running visualizer.")
#v.run(interval=10)
