import numpy as np
from PIL import Image

# Generate a mean filter kernel
def mean_filter_kernel(filter_size):
    return np.ones((filter_size, filter_size)) / (filter_size * filter_size)

# Generate a gaussian filter kernel
def gaussian_filter_kernel(filter_size, sigma):
    # 1. Calculate the Center Position: The center position of the Gaussian kernel is (filter_size - 1) / 2.
    # For a 3x3 Gaussian kernel, the center position is (1, 1).
    # 2. Generate the Coordinate Grid: Use the np.fromfunction function to generate a coordinate grid,
    # where x and y represent the coordinates of each element.
    kernel = np.fromfunction(
        lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(- ((x - (filter_size-1)/2)**2 + (y - (filter_size-1)/2)**2) / (2 * sigma**2)),
        (filter_size, filter_size)
    )
    return kernel / np.sum(kernel) # Normalization

# Load the image
img = Image.open('../Reference/data/tower.jpg').convert('RGB')
img_array = np.array(img)

# Define kernels
gaussian_kernel = gaussian_filter_kernel(7, 1.0)
mean_kernel = mean_filter_kernel(3)

# Apply a filter to the image
def apply_filter(image_np, kernel):
    filter_size = kernel.shape[0]
    # To add a border around the image to handle edge pixels during filtering.
    padded_image = np.pad(image_np, ((filter_size // 2, filter_size // 2), (filter_size // 2, filter_size // 2), (0, 0)),
                          mode='edge')

    height, width, channels = img_array.shape
    filtered_image = np.zeros_like(image_np)
    for channel in range(channels):
        for col in range(width):
            for row in range(height):
                filtered_image[row, col, channel] = np.sum(
                    kernel * padded_image[row:row + filter_size, col:col + filter_size, channel])

    return filtered_image

gaussian_filtered_image = apply_filter(img_array, gaussian_kernel)
gaussian_result_image = Image.fromarray(gaussian_filtered_image)
gaussian_result_image.show()

mean_filtered_image = apply_filter(img_array, mean_kernel)
mean_result_image = Image.fromarray(mean_filtered_image)
mean_result_image.show()