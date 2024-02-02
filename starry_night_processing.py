import cv2
import numpy as np


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


# Filter edges to connect them with erosion then dilation
def connect_edges(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)


def main():
    image_path = load_image()
    cv2.imshow("Original Image", image_path)

    preprocessed_image = image_preprocessing(image_path)
    cv2.imshow("Preprocessed Image", preprocessed_image)

    edge_detected_image = edge_detection(preprocessed_image)
    cv2.imshow("Edge Detected Image", edge_detected_image)

    connected_edges_image = connect_edges(edge_detected_image)
    cv2.imshow("Connected Edges Image", connected_edges_image)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
