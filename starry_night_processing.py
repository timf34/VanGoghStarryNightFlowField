import cv2
import numpy as np

from typing import List, Dict

# Load the images
image_path = 'images/StarryNight.jpg'  # Update this path
image = cv2.imread(image_path)

# Resize the image
image = cv2.resize(image, (800, 500))

if image is None:
    print(f"Error: Unable to load image at {image_path}")
else:
    # Proceed with processing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (3, 3), 0)

cv2.imshow("Gray", gray)
# Show blurred image
# cv2.imshow('Blurred', blurred)

edges = cv2.Canny(gray, threshold1=100, threshold2=200)

# Display the edges
cv2.imshow('Edges', edges)

# Compute the gradient in x and y directions
grad_x = cv2.Sobel(edges, cv2.CV_64F, 1, 0, ksize=5)
grad_y = cv2.Sobel(edges, cv2.CV_64F, 0, 1, ksize=5)

# Display the gradients
cv2.imshow('Gradient X', grad_x)
cv2.imshow('Gradient Y', grad_y)

# Compute the direction of the gradient
angle = np.arctan2(grad_y, grad_x)

def draw_vectors(image, angle, step=10):
    for y in range(0, image.shape[0], step):
        for x in range(0, image.shape[1], step):
            dx = int(np.cos(angle[y, x]) * step)
            dy = int(np.sin(angle[y, x]) * step)
            cv2.arrowedLine(image, (x, y), (x + dx, y + dy), (255, 0, 0), 1)
    cv2.imshow('Vector Field', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Visualize the vector field on a copy of the original image
image_copy = image.copy()
draw_vectors(image_copy, angle)
