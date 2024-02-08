import cv2
import numpy as np
import pygame


class VectorField:
    def __init__(self, edge_image):
        self.edge_image = edge_image
        self.height, self.width = edge_image.shape
        self.vector_field = np.zeros((self.height, self.width, 2))  # For storing vector (angle) at each point
        # Compute the distance transform and gradients
        self.compute_distance_transform()

    def compute_distance_transform(self):
        self.distance_transform = cv2.distanceTransform(255 - self.edge_image, cv2.DIST_L2, 5)
        self.grad_x = cv2.Sobel(self.distance_transform, cv2.CV_64F, 1, 0, ksize=5)
        self.grad_y = cv2.Sobel(self.distance_transform, cv2.CV_64F, 0, 1, ksize=5)

    def create_uniform_points(self):
        for y in range(0, self.height):
            for x in range(0, self.width):
                self.set_vector_at_point(x, y)

    def set_vector_at_point(self, x, y):
        grad_x_at_point = self.grad_x[y, x]
        grad_y_at_point = self.grad_y[y, x]
        norm = np.sqrt(grad_x_at_point**2 + grad_y_at_point**2) + 1e-6
        self.vector_field[y, x, :] = [grad_x_at_point / norm, grad_y_at_point / norm]

    def visualize_vector_field_with_pygame(self):
        # Initialize Pygame window
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Vector Field Visualization")

        # Main loop
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((0, 0, 0))  # Fill screen with black

            # Draw vectors
            for y in range(0, self.height, 10):  # Adjust the step for sparser vector field
                for x in range(0, self.width, 10):
                    vx, vy = self.vector_field[y, x]
                    end_point = (int(x + vx * 10), int(y + vy * 10))  # Scale vector for visibility
                    pygame.draw.line(screen, (255, 255, 255), (x, y), end_point)

            pygame.display.flip()  # Update the full display Surface to the screen

        pygame.quit()


def load_image(image_path='images/StarryNight.jpg') -> np.ndarray:
    image = cv2.imread(image_path)
    image = cv2.resize(image, (800, 600))
    return image


def image_preprocessing(image: np.ndarray) -> np.ndarray:
    # sourcery skip: inline-immediately-returned-variable
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return blurred


def edge_detection(image: np.ndarray) -> np.ndarray:
    return cv2.Canny(image, 30, 150)


def main():
    # image_path = load_image()
    # cv2.imshow("Original Image", image_path)
    #
    # preprocessed_image = image_preprocessing(image_path)
    # cv2.imshow("Preprocessed Image", preprocessed_image)
    #
    # edge_detected_image = edge_detection(preprocessed_image)
    # cv2.imshow("Edge Detected Image", edge_detected_image)
    image = np.zeros((500, 500), dtype=np.uint8)

    # Fill one half along the diagonal with 1s
    for i in range(500):
        for j in range(500):
            if i == j:
                image[i, j] = 255

    # Create and visualize the vector field based on the edge-detected random noise
    cv2.imshow("image", image)
    vector_field_creator = VectorField(image)
    vector_field_creator.create_uniform_points()
    vector_field_creator.visualize_vector_field_with_pygame()

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
