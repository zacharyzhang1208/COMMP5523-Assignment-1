from PIL import Image
import numpy as np

# Load the image
img = Image.open('../Reference/data/2.2_gray_smoothed.jpg')
smoothed_image_array = np.array(img, dtype=np.float32)  # Convert the grayscale image to a NumPy array with a specific data type

# Define Sobel operator kernels
sobel_x_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])

sobel_y_kernel = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])

# Apply Sobel operator to get x-gradient and y-gradient
def apply_sobel_operator(image, kernel):
    filter_size = kernel.shape[0]
    padded_image = np.pad(image, ((filter_size // 2, filter_size // 2), (filter_size // 2, filter_size // 2)), mode='edge')
    height, width = image.shape
    gradient_image = np.zeros_like(image)
    for col in range(width):
        for row in range(height):
            gradient_image[row, col] = np.sum(kernel * padded_image[row:row + filter_size, col:col + filter_size])
    return gradient_image

sobel_x = apply_sobel_operator(smoothed_image_array, sobel_x_kernel)
sobel_y = apply_sobel_operator(smoothed_image_array, sobel_y_kernel)

# Calculate the magnitude of the gradient G = root(Gx^2 + Gy^2)
gradient_magnitude = np.hypot(sobel_x, sobel_y)

# Normalize the gradients to the range [0, 255]
sobel_x_uint8 = np.uint8(255 * (sobel_x - np.min(sobel_x)) / (np.max(sobel_x) - np.min(sobel_x)))
sobel_y_uint8 = np.uint8(255 * (sobel_y - np.min(sobel_y)) / (np.max(sobel_y) - np.min(sobel_y)))
print(np.max(gradient_magnitude) - np.min(gradient_magnitude))
gradient_magnitude_uint8 = np.uint8(255 * (gradient_magnitude - np.min(gradient_magnitude)) / (np.max(gradient_magnitude) - np.min(gradient_magnitude)))

# Save the x-gradient, y-gradient, and magnitude images
Image.fromarray(sobel_x_uint8).save('../Reference/data/2.3_G_x.jpg')
Image.fromarray(sobel_y_uint8).save('../Reference/data/2.3_G_y.jpg')
Image.fromarray(gradient_magnitude_uint8).save('../Reference/data/2.3_G.jpg')
