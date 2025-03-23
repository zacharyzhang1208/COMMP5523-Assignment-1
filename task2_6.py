import numpy as np
from PIL import Image

# Load the image
img = Image.open('../Reference/data/2.5_edgemap.jpg')
suppress = np.array(img, dtype=np.uint8)

# Perform edge detection using a simple method (e.g., Sobel operator)
def sobel_edge_detection(image):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    Ix = np.zeros_like(image)
    Iy = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            Ix[i, j] = np.sum(Kx * image[i-1:i+2, j-1:j+2])
            Iy[i, j] = np.sum(Ky * image[i-1:i+2, j-1:j+2])
    G = np.hypot(Ix, Iy)
    G = (G / G.max() * 255).astype(np.uint8)
    return G

edges = sobel_edge_detection(suppress)

# Perform Hough Transform
def hough_transform(image):
    height, width = image.shape
    max_dist = int(np.hypot(height, width))
    accumulator = np.zeros((2 * max_dist, 180), dtype=np.int32)
    for y in range(height):
        for x in range(width):
            if image[y, x] > 0:  # Edge pixel
                for theta in range(180):
                    rho = int(x * np.cos(np.deg2rad(theta)) + y * np.sin(np.deg2rad(theta)))
                    accumulator[rho + max_dist, theta] += 1
    return accumulator

accumulator = hough_transform(edges)

# Normalize and save the accumulator image
accumulator_image = (accumulator / accumulator.max() * 255).astype(np.uint8)
Image.fromarray(accumulator_image).save('../Reference/data/2.6_hough.jpg')

print("Hough Transform voting accumulator saved as data/2.6_hough.jpg")