import os
from datetime import datetime
import logging

from simulation.Simulation import Simulation
from simulation.Victim import Victim

# Static variables. These should not change.
logger = logging.getLogger(__name__)
logging.basicConfig(format='\033[1;31m[%(levelname)s]\033[0m \033[33m%(name)s\033[0m \033[94m%(funcName)s\033[0m - %(message)s', level=logging.INFO)
proj_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(proj_dir,"resources/settings.json")

# Tuning parameters.
# These can change, and will likely be determined by some 'randomizer' later.
lat = 30.0
lon = -80.0
margin = 5 # miles
start_date = datetime(2023, 1, 1, 00, 00, 00)
end_date = datetime(2023, 1, 5, 00,00,00)

logger.info("\033[32mStarting application.\033[0m")
logger.info("Parameters are predefined in __main__ for stability; this is not a final product.")
logger.info("Simulation Parameters: Center=(%s, %s), Margin=%s miles, Start=%s, End=%s",
            lat, lon, margin, start_date.strftime('%d%b%Y'), end_date.strftime('%d%b%Y')
        )

logger.info("\033[32mRunning simulation.\033[0m")
s = Simulation(lat, lon, config_path, start_date, end_date)
v=Victim(10,10,10, 30.1, -80.2, config_path)
s._add_victim(v)
s.RunShow()



