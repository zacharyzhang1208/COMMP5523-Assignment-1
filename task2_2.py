from PIL import Image
import numpy as np
from scipy import ndimage

img = Image.open('../Reference/data/2.1_gray.jpg')
gray_image_array = np.array(img, dtype=np.uint8) # Convert the grayscale image to a NumPy array with a specific data type

# Perform Gaussian smoothing
sigma = 1.0  # We can adjust this value to see how it affects the results
smoothed_image = ndimage.gaussian_filter(gray_image_array, sigma=sigma)
smoothed_result = Image.fromarray(gray_image_array)
# smoothed_result.show()
smoothed_result.save('../Reference/data/2.2_gray_gaussian.jpg')