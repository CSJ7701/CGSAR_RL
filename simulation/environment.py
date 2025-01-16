"""Defines the Environment class."""


class Environment:
    """Test environment, with obstacles."""

    def __init__(self, width, height):
        """Initialize an instance of the 'Environemnt' class."""
        self.width = width
        self.height = height
        self.obstacles = []

    def add_obstacle(self, x, y):
        """Add an obstacle to the environment."""
        self.obstacles.append((x, y))

    def is_collision(self, x, y):
        """Check whether a position collides with an existing obstacle."""
        return (x, y) in self.obstacles
