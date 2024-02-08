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
        angle += np.pi / 2  # Rotate by 90 degrees

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

            # Adjust this loop to iterate over every point
            for y in range(0, self.height, 10):
                for x in range(0, self.width, 10):
                    vx, vy = self.vector_field[y, x]
                    # Scale the vectors for better visibility if necessary
                    scale = 10  # Adjust scale factor as needed for visibility
                    end_point = (int(x + vx * scale), int(y + vy * scale))
                    # Draw only if vector is non-zero
                    # if vx != 0 or vy != 0:
                    pygame.draw.line(screen, (255, 255, 255), (x, y), end_point)

            pygame.display.flip()

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
    image_path = load_image()
    cv2.imshow("Original Image", image_path)

    preprocessed_image = image_preprocessing(image_path)
    cv2.imshow("Preprocessed Image", preprocessed_image)

    edge_detected_image = edge_detection(preprocessed_image)
    cv2.imshow("Edge Detected Image", edge_detected_image)
    # image = np.zeros((500, 500), dtype=np.uint8)
    #
    # # Fill one half along the diagonal with 1s
    # for i in range(500):
    #     for j in range(500):
    #         if i == j:
    #             image[i, j] = 255

    # Create and visualize the vector field based on the edge-detected random noise
    # cv2.imshow("image", edge_detected_image)
    vector_field_creator = VectorField(edge_detected_image)
    vector_field_creator.visualize_vector_field_with_pygame()

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
