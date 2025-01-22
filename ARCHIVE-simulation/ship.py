"""Defines the Ship class."""

import math


class Ship:
    """Defines the 'Ship' class."""

    def __init__(self, x, y, heading=0, speed=0):
        """Define an instance of the 'Ship' class."""
        self.x = x
        self.y = y
        self.heading = heading  # Degrees.
        self.speed = speed      # Units per update.  TODO: Scale.
        # Drag
        # Accel rate
        # Turn Radius

    def update_position(self, dt):
        """Check the ship's current position based on speed and heading."""
        # Convert heading degrees to radians
        rad = math.radians(self.heading)
        self.x += self.speed * math.cos(rad) * dt
        self.y += self.speed * math.sin(rad) * dt

    def turn(self, angle):
        """Change the ship's current heading."""
        self.heading = (self.heading + angle) % 360

    def set_speed(self, speed):
        """Set the ship's current speed."""
        self.speed = speed
