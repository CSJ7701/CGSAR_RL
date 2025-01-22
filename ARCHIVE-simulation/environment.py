"""Defines the Environment class."""
import math
from map_data import map_data


class Environment:
    """Test environment, with obstacles."""

    def __init__(self, lat, lon, width, height, margin_miles=5):
        """Initialize an instance of the 'Environemnt' class."""
        self.width = width
        self.height = height
        self.margin_miles = margin_miles
        self.coastlines = []

        # Compute lat/lon bounds
        self.lat_min, self.lat_max, self.lon_min, self.lon_max = self.calculate_bounds(
            lat, lon, margin_miles
        )

        # Initialize the map_data
        self.map_data = map_data(
            self.lat_min, self.lat_max, self.lon_min, self.lon_max, width, height
        )
        self.load_coastlines()

    def calculate_bounds(self, lat, lon, margin_miles):
        """Calculate latitude/longitude bounds based on a center coordinate and a margin."""
        print("Calculating environment bounds...")
        # Approximation: 1 degree latitude ~ 69 miles
        # 1 degree longitude varied based on latitude, use cosine adjustment
        lat_margin = margin_miles / 69.0
        lon_margin = margin_miles / (69.0 * math.cos(math.radians(lat)))

        lat_min = lat - lat_margin
        lat_max = lat + lat_margin
        lon_min = lon - lon_margin
        lon_max = lon + lon_margin
        return lat_min, lat_max, lon_min, lon_max

    def load_coastlines(self):
        """Fetch and load coastline data into the env."""
        print("Loading map data...")
        self.map_data.fetch_coastlines()
        self.coastlines = self.map_data.coastlines
        print("Map data fully loaded.")

    def is_collision(self, x, y):
        """Check whether a position collides with an obstacle or coastline."""
        for segment in self.coastlines:
            for sx, sy in segment:
                if abs(x - sx) < 5 and abs(y-sy) < 5:  # Collision radius
                    return True
        return False
