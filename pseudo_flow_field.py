"""
File to get our vector field creation from an edge image correct (or what we want at least)
"""
import cv2
import numpy as np
import pygame


class VectorField:
    def __init__(self, edge_image):
        self.edge_image = edge_image
        self.height, self.width = edge_image.shape
        self.vector_field = np.zeros((self.height, self.width, 2))  # For storing vector (direction) at each point
        self.compute_gradients()

    def compute_gradients(self):
        # Use Sobel operator to find gradients
        grad_x = cv2.Sobel(self.edge_image, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(self.edge_image, cv2.CV_64F, 0, 1, ksize=5)

        # Calculate gradient magnitude and angle
        magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
        angle = np.arctan2(grad_y, grad_x)

        # Use a fixed lower threshold for edge detection
        edge_threshold = 500  # Example threshold, adjust as needed

        for y in range(self.height):
            for x in range(self.width):
                # Directly use gradients where the magnitude exceeds a minimal threshold
                # print(magnitude[y, x])
                # print(angle[y, x])
                if magnitude[y, x] > edge_threshold:
                    self.vector_field[y, x, :] = [np.cos(angle[y, x]), np.sin(angle[y, x])]
                else:
                    self.vector_field[y, x, :] = [0, 0]  # Set vector to null if not near an edge
                # print(self.vector_field[y, x, :], "vector")

    def visualize_vector_field_with_pygame(self):
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Vector Field Visualization")

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((0, 0, 0))  # Fill screen with black

            for y in range(0, self.height, 10):
                for x in range(0, self.width, 10):
                    vx, vy = self.vector_field[y, x]
                    end_point = (int(x + vx * 10), int(y + vy * 10))
                    pygame.draw.line(screen, (255, 255, 255), (x, y), end_point)

            pygame.display.flip()

        pygame.quit()


def main():
    image = np.zeros((500, 500), dtype=np.uint8)

    # Create a diagonal edge for demonstration
    for i in range(250):
        for j in range(250):
            if i == j:
                image[i, j] = 255

    cv2.imshow("image", image)

    vector_field_creator = VectorField(image)
    vector_field_creator.visualize_vector_field_with_pygame()


if __name__ == "__main__":
    main()
