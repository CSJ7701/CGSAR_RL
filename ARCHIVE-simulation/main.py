import pygame
from ship import Ship
from environment import Environment

pygame.init()

WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

center_lat, center_lon = 40.7128, -74.0060

# Initialize the game environment with center coords.
environment = Environment(center_lat, center_lon, WIDTH, HEIGHT)
ship = Ship(400, 300, heading=90)



running = True
# Main Loop
while running:
    dt = clock.tick(30) / 1000  # Time step for updates - 30 FPS

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Keyboard Input
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT]:
        ship.turn(-5)
    if keys[pygame.K_RIGHT]:
        ship.turn(5)
    if keys[pygame.K_UP]:
        ship.set_speed(100)
    if keys[pygame.K_DOWN]:
        ship.set_speed(0)

    ship.update_position(dt)
    if environment.is_collision(int(ship.x), int(ship.y)):
        print("Collision detected!")

    # Draw background
    screen.fill((0, 0, 0))

    # Draw coastlines
    for segment in environment.coastlines:
        if len(segment) > 1:
            pygame.draw.lines(screen, (0, 0, 255), False, segment, 2)

        # Draw ship
    pygame.draw.circle(screen, (0, 255, 0), (int(ship.x), int(ship.y)), 5)
    pygame.display.flip()
    
pygame.quit()
