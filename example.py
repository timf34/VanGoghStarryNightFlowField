import pygame
import math

class VectorField:
    def __init__(self, width, height, resolution):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.nx = round(width / resolution)
        self.ny = round(height / resolution)
        self.grid = [[0 for _ in range(self.ny)] for _ in range(self.nx)]  # Assuming defaultAngle is 0

    def clone(self):
        copy = VectorField(self.width, self.height, self.resolution)
        copy.grid = [row[:] for row in self.grid]
        return copy

    def get_cell(self, ix, iy):
        ix = min(self.nx - 1, max(0, ix))
        iy = min(self.ny - 1, max(0, iy))
        return self.grid[ix][iy]

    def set_cell(self, ix, iy, angle):
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            self.grid[ix][iy] = angle

    def get_cell_index(self, x, y):
        return (x // self.resolution, y // self.resolution)

    def get_field(self, x, y):
        ix, iy = self.get_cell_index(x, y)
        alphax = (x % self.resolution) / self.resolution
        alphay = (y % self.resolution) / self.resolution

        # Placeholder for angleLerp function
        def angle_lerp(a1, a2, t):
            return a1 + (a2 - a1) * t

        return angle_lerp(
            angle_lerp(self.get_cell(ix, iy), self.get_cell(ix + 1, iy), alphax),
            angle_lerp(self.get_cell(ix, iy + 1), self.get_cell(ix + 1, iy + 1), alphax),
            alphay
        )

    def draw(self, screen):
        for i in range(self.nx):
            for j in range(self.ny):
                start_pos = (i * self.resolution + self.resolution // 2, j * self.resolution + self.resolution // 2)
                angle = self.get_cell(i, j)
                end_pos = (start_pos[0] + math.cos(angle) * self.resolution * 0.8, start_pos[1] + math.sin(angle) * self.resolution * 0.8)
                pygame.draw.line(screen, pygame.Color('white'), start_pos, end_pos)


def main():
    pygame.init()
    width, height = 800, 600
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Vector Field Demo")

    clock = pygame.time.Clock()
    vector_field = VectorField(width, height, 40)  # Example resolution set to 40

    # Example: Set a diagonal gradient for demonstration
    for i in range(vector_field.nx):
        for j in range(vector_field.ny):
            angle = (i / vector_field.nx) * 2 * math.pi
            vector_field.set_cell(i, j, angle)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))  # Fill the screen with black
        vector_field.draw(screen)
        pygame.display.flip()

        clock.tick(60)  # Limit to 60 frames per second

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
