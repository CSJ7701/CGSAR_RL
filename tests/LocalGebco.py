import rasterio
from rasterio.windows import Window
from rasterio.transform import from_origin
import numpy as np
import pygame
from ship import Ship
#import matplotlib
#import matplotlib.pyplot as plt
#matplotlib.use('TkAgg')

file_path = "../resources/GEBCO_tiff/gebco_2024_n35.0_s25.0_w-82.0_e-75.0.tif"
lat, lon = 30.0, -80.0
NAV_DRAFT_FEET = 100
NAV_DRAFT_METERS = NAV_DRAFT_FEET * 0.3048

with rasterio.open(file_path) as dataset:
    print("CRS: ",dataset.crs)
    print("Bounds: ",dataset.bounds)
    print("Resolution: ",dataset.res)
    print("Shape: ", (dataset.width, dataset.height))
    print("===========================")

    row, col = dataset.index(lon, lat)
    print(f"Pixel coordinates: row={row}, col={col}")

    window_size = 500
    window = Window(col_off=col - window_size, row_off=row - window_size, width=2 * window_size, height=2 * window_size)

    data = dataset.read(1, window=window)

    transform = dataset.window_transform(window)
    print("Window Bounds: \n", transform)

# print(data)
    
# plt.imshow(data, cmap="viridis", extent=(lon-0.1, lon+0.1, lat-0.1, lat+0.1))
# plt.colorbar(label="Depth (meters)")
# plt.title("DATA?")
# plt.xlabel("Long")
# plt.ylabel("Lat")
# plt.show()

def check_collision(ship_x, ship_y, elevation_data, draft=NAV_DRAFT_METERS):
    """
    Check if the ship has grounded based on its draft.
    :param ship_x: Ship's x coord
    :param ship_y: Ship's y coord
    :param elevation_data: 2D numpy array of elevation values.
    :param draft: Nav draft for the ship in meter's.
    :return: True if grounded, false otherwise.
    """
    if 0 <= ship_x < elevation_data.shape[1] and 0 < ship_y < elevation_data.shape[0]:
        depth = elevation_data[ship_y, ship_x]
        return depth >= -draft
    return False

def draw_nav_draft(surface, elevation_data, draft=NAV_DRAFT_METERS, color=(255,100,20)):
    """
    Draw a safety line for the navigational draft.
    :param surface: Pygame surface to draw on.
    :param elevation_data: 2D numpy array of elevation values.
    :param draft: Navigational draft in meters.
    :param color: Color of the safety line (default: red).
    """
    width, height = surface.get_size()
    for x in range(width):
        for y in range(height):
            if elevation_data[y,x] >= -draft:
                neighbors = [
                    (x-1,y), (x+1,y),
                    (x,y-1), (x,y+1)
                ]
                for nx, ny in neighbors:
                    if 0<=nx<width and 0<=ny<height:
                        if elevation_data[ny,nx] < -draft:
                            surface.set_at((x,y), color)
                            break
                


def normalize_data(data, vmin=None, vmax=None):
    """Normalize to 0-255 colormap"""
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    normalized = (data-vmin) / (vmax-vmin)
    return (normalized * 255).astype(np.uint8)

def create_surface(data):
    """Convert normalized color map to a pygame surface."""
    trans=np.transpose(data)
    print(trans)
    colorized = np.zeros((trans.shape[0], trans.shape[1],3), dtype=np.uint8)
    #    colorized = np.stack([data,data,data], axis=-1)

    # Blue tint for water, depth <= 0
    water_mask = trans <= 0
    colorized[water_mask] = [0,128,255] # Blue

    # Green for land (depth > 0)
    land_mask = trans > 0
    colorized[land_mask] = [240,226,182] # Tan
    
    surface = pygame.surfarray.make_surface(colorized)
    # surface = pygame.transform.rotate(surface, 90)
    return surface

def display_pygame_window(data):
    """Display the normalized data in pygame."""
    pygame.init()
    normalized = normalize_data(data)
    surface = create_surface(normalized)

    screen = pygame.display.set_mode(surface.get_size())
    pygame.display.set_caption("Elevation Data")

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            screen.blit(surface, (0,0))
            pygame.display.flip()
    pygame.quit()

def display_pygame_with_collision(elevation_data, ship_position, draft=NAV_DRAFT_METERS):
    """Display the elevation data with collision logic and a safety line."""
    pygame.init()
    clock = pygame.time.Clock()

    # Normalize data
    #normalized_data = normalize_data(elevation_data)

    # Create surface
    #surface = create_surface(normalized_data)
    surface = create_surface(elevation_data)

    # Initialize screen
    screen = pygame.display.set_mode(surface.get_size())
    dt = clock.tick(30) / 1000
    pygame.display.set_caption("GEBCO Simulation with Collision Logic")

    # Ship
    ship = Ship(ship_position[0], ship_position[1], heading=90)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            ship.turn(-5)
        if keys[pygame.K_RIGHT]:
            ship.turn(5)
        if keys[pygame.K_UP]:
            ship.set_speed(500)
        if keys[pygame.K_DOWN]:
            ship.set_speed(0)

        ship.update_position(dt)

        # Check for collision
        grounded = check_collision(int(ship.x), int(ship.y), elevation_data, draft)


        # Blit the surface and draw the safety line
        screen.blit(surface, (0, 0))
        draw_nav_draft(screen, elevation_data, draft)

        # Draw the ship (different color if grounded)
        ship_color = (255, 0, 0) if grounded else (255, 255, 0)
        pygame.draw.circle(screen, ship_color, (int(ship.x), int(ship.y)), 5)

        pygame.display.flip()

    pygame.quit()

# Test the elevation display in pygame
#display_pygame_window(data)
    
# Test with collision
ship_position = (data.shape[1]//2, data.shape[0]//2)
display_pygame_with_collision(data, ship_position)
