import overpy
import pygame
from pyproj import Transformer

class map_data:
    """Fetch and process geographical data for the environment class."""

    def __init__(self, lat_min, lat_max, lon_min, lon_max, width, height):
        """Initialize map_data with map bounds and dimensions."""
        self.api = overpy.Overpass()
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.w = width
        self.h = height
        self.coastlines = []

        # Coordinate transformer: WGS84 (lat/lon) -> screen coordinates
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857")

    def fetch_coastlines(self):
        """Fetch coastline data using Overpass API."""
        query = f"""
        [out:json];
        way["natural"="coastline"]({self.lat_min},{self.lon_min},{self.lat_max},{self.lon_max});
        (._;>;);
        out body;
        """
        try:
            result = self.api.query(query)
            print(f"Fetched {len(result.ways)} coastline ways.")
            self.process_coastlines(result.ways)
        except Exception as e:
            print(f"Error fetching data: {e}")

    def process_coastlines(self, ways):
        """Convert fetched coastlines to screen coordinates."""
        self.coastlines = []
        for way in ways:
            coords = [(float(node.lat), float(node.lon)) for node in way.nodes]
            screen_coords = [self.to_screen(lat, lon) for lat, lon in coords]
            self.coastlines.append(screen_coords)

    def to_screen(self, lat, lon):
        """Convert geographical coordinates to screen space."""
        # Transform lat/lon to Mercator projection (EPSG:3857)
        x, y = self.transformer.transform(lat, lon)
        # Normalize
        x_norm = (x - self.lon_min) / (self.lon_max - self.lon_min) * self.w
        y_norm = ((y - self.lat_min) / (self.lat_max - self.lat_min) * self.h)
        y_norm = self.h - y_norm
        return int(x_norm), int(y_norm)
