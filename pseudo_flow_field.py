import numpy as np
import cv2
import pygame
import math

class VectorField:
    def __init__(self, edge_image):
        self.edge_image = edge_image
        self.height, self.width = edge_image.shape[:2]
        self.vector_field = np.zeros((self.height, self.width, 2))  # For storing vector (direction) at each point
        self.compute_gradients()

    def compute_gradients(self):
        # Use Sobel operator to find gradients
        grad_x = cv2.Sobel(self.edge_image, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(self.edge_image, cv2.CV_64F, 0, 1, ksize=5)

        # Calculate gradient magnitude and angle
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)

        # Adjusting the angle to align with the edge direction
        angle += np.pi / 2  # Rotate by 90 degrees

        # Use a fixed lower threshold for edge detection
        edge_threshold = 100  # Lowered threshold for better sensitivity

        # Populate the vector field based on gradient information
        for y in range(self.height):
            for x in range(self.width):
                if magnitude[y, x] > edge_threshold:
                    self.vector_field[y, x] = [np.cos(angle[y, x]), np.sin(angle[y, x])]
                else:
                    self.vector_field[y, x] = [0, 0]  # Null vector if below threshold

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

            # Visualize vectors
            for y in range(0, self.height, 10):  # Adjust step for performance/visibility
                for x in range(0, self.width, 10):
                    vx, vy = self.vector_field[y, x]
                    scale = 5  # Adjust for visibility
                    end_point = (int(x + vx * scale), int(y + vy * scale))
                    pygame.draw.line(screen, (255, 255, 255), (x, y), end_point)

            pygame.display.flip()

        pygame.quit()

def main():
    # Simple edge for demonstration
    image = np.zeros((500, 500), dtype=np.uint8)
    cv2.line(image, (0, 0), (500, 500), 255, 10)  # Diagonal line with thickness

    # Display the original image
    cv2.imshow("Edge Image", image)
    # cv2.waitKey(0)  # Wait for a key press to close
    # cv2.destroyAllWindows()

    vector_field_creator = VectorField(image)
    vector_field_creator.visualize_vector_field_with_pygame()

if __name__ == "__main__":
    main()
