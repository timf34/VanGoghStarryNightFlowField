import pygame
import numpy as np
import random
import math
from noise import pnoise2

# Initialize Pygame
pygame.init()

# Screen dimensions
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Flow Field with Perlin Noise")

# Colors
white = (255, 255, 255)
black = (0, 0, 0)

# Flow field resolution
resolution = 20
cols, rows = width // resolution, height // resolution

# Perlin noise parameters
# Adjust these parameters to change the noise characteristics
perlin_scale = 0.1  # Scales the input coordinates. Smaller values will make the noise smoother.
octaves = 5  # The number of passes, more passes mean more detail and longer calculation time.
persistence = 0.5  # Amplitude of each successive octave relative to the one before it.
lacunarity = 2.0  # Frequency of each successive octave relative to the one before it.


# VectorField class
class VectorField:
    def __init__(self):
        self.field = np.zeros((cols, rows, 2), dtype=np.float64)

    def generate(self):
        for i in range(cols):
            for j in range(rows):
                angle = perlin_noise(i, j) * 2 * math.pi
                self.field[i, j] = [np.cos(angle), np.sin(angle)]

    def lookup(self, position):
        column = int(position[0] / resolution)
        row = int(position[1] / resolution)
        column = np.clip(column, 0, cols - 1)
        row = np.clip(row, 0, rows - 1)
        return self.field[column, row]

    def draw(self):
        for i in range(cols):
            for j in range(rows):
                angle = math.atan2(self.field[i, j][1], self.field[i, j][0])
                x0 = i * resolution + resolution // 2
                y0 = j * resolution + resolution // 2
                x1 = x0 + math.cos(angle) * 10
                y1 = y0 + math.sin(angle) * 10
                pygame.draw.line(screen, white, (x0, y0), (x1, y1), 1)

# Particle class
class Particle:
    def __init__(self):
        self.position = np.random.rand(2) * [width, height]
        angle = random.uniform(0, 2 * math.pi)
        self.velocity = np.array([math.cos(angle), math.sin(angle)]) * (random.random() * 2 - 1)
        self.acceleration = np.zeros(2)
        self.max_speed = random.uniform(0.7, 3)

    def update(self):
        self.velocity += self.acceleration
        self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed)
        self.position += self.velocity
        self.acceleration *= 0
        # Occasionally apply a small random force
        if random.randint(0, 50) == 0:  # Adjust frequency as needed
            self.apply_force(np.random.rand(2) * 0.5 - 0.25)  # Small random force

    def apply_force(self, force):
        self.acceleration += force

    def show(self):
        pygame.draw.circle(screen, white, self.position.astype(int), 2)

    def edges(self):
        if self.position[0] > width: self.position[0] = 0
        if self.position[0] < 0: self.position[0] = width
        if self.position[1] > height: self.position[1] = 0
        if self.position[1] < 0: self.position[1] = height


def perlin_noise(x, y):
    # The noise function expects 3D input (x, y, z), where z is usually time or a constant
    noise_val = pnoise2(x * perlin_scale,
                        y * perlin_scale,
                        octaves=octaves,
                        persistence=persistence,
                        lacunarity=lacunarity,
                        repeatx=width,
                        repeaty=height,
                        base=0)
    return noise_val


def main():
    vector_field = VectorField()
    vector_field.generate()

    particles = [Particle() for _ in range(1000)]  # Create 100 particles

    clock = pygame.time.Clock()  # For controlling the frame rate

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(black)  # Clear the screen

        vector_field.draw()  # Draw the vector field

        # Update and show particles
        for particle in particles:
            force = vector_field.lookup(particle.position)  # Get the vector from the field
            particle.apply_force(force)  # Apply this force to the particle
            particle.update()  # Update particle's position
            particle.edges()  # Check for screen edges and wrap around if necessary
            particle.show()  # Draw the particle on the screen

        pygame.display.flip()  # Update the full display Surface to the screen

        clock.tick(60)  # Limit the frame rate to 60 frames per second

    pygame.quit()  # Uninitialize all pygame modules and quit the program


if __name__ == "__main__":
    main()


