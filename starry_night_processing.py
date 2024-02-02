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


def filter_long_edges(edges: np.ndarray, min_length: float) -> np.ndarray:
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    long_edges = np.zeros_like(edges)
    # Filter contours by length
    for contour in contours:
        if cv2.arcLength(contour, False) > min_length:
            cv2.drawContours(long_edges, [contour], -1, 255, thickness=1)
    return long_edges


def main():
    image_path = load_image()
    cv2.imshow("Original Image", image_path)

    preprocessed_image = image_preprocessing(image_path)
    cv2.imshow("Preprocessed Image", preprocessed_image)

    edge_detected_image = edge_detection(preprocessed_image)
    cv2.imshow("Edge Detected Image", edge_detected_image)

    long_edge_image = filter_long_edges(edge_detected_image, min_length=100)
    cv2.imshow("Long Edge Image", long_edge_image)

    cv2.waitKey(0)


if __name__ == "__main__":
    main()
