import overpy
import math
import pygame

width = 800
height = 800

center = (41.326231, -72.120765)

min_lat = 41.263872 
min_lon = -72.219127
max_lat = 41.388529
max_lon = -72.022403

def calculate_bounds(lat, lon, margin_miles):
        # Approximation: 1 degree latitude ~ 69 miles
        # 1 degree longitude varied based on latitude, use cosine adjustment
        lat_margin = margin_miles / 69.0
        lon_margin = margin_miles / (69.0 * math.cos(math.radians(lat)))

        lat_min = lat - lat_margin
        lat_max = lat + lat_margin
        lon_min = lon - lon_margin
        lon_max = lon + lon_margin
        return lat_min, lat_max, lon_min, lon_max

def to_screen(lat, lon):
    try:
        norm_lat = (lat - min_lat) / (max_lat - min_lat)
        norm_lon = (lon - min_lon) / (max_lon - min_lon)
    except ZeroDivisionError:
        print("ERROR... dimensions are 0. Check data.")
        return 0,0

    screen_x = int(norm_lon * width)
    screen_y = int((1-norm_lat) * height)

    return screen_x, screen_y

# print(calculate_bounds(center[0], center[1], 5))

coastlines = []
api = overpy.Overpass()
query = f"""
        [out:json];
        way["natural"="coastline"]({min_lat},{min_lon},{max_lat},{max_lon});
        (._;>;);
        out body;
        """
result = api.query(query)
ways = result.ways

for way in ways:
    coords = [(float(node.lat), float(node.lon)) for node in way.nodes]
    screen_coords = [to_screen(lat,lon) for lat,lon in coords]
    coastlines.append(screen_coords)

pygame.init()
screen = pygame.display.set_mode((width,height))
running = True
while running:
    for segment in coastlines:
        if len(segment) > 1:
            pygame.draw.lines(screen, (0,0,255), False, segment, 2)


    pygame.display.flip()
pygame.quit()
    

