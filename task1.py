from PIL import Image
import numpy as np

img = Image.open('../Reference/data./library.jpg').convert("RGB") # read images
img_array = np.array(img)

# img.show()

# define 3*3 gaussian filter and filter_size
filter_size = 3
gaussian_filter = np.array([[1, 4, 1],
                           [2, 4,2],
                           [1, 2, 1]]) / 16.0


height, width, channels = img_array.shape
# print(img_array.shape)

# Create an output array to store the blurred image
blurred_image_array = np.zeros_like(img_array)

for channel in range(channels):
    for i in range(height - filter_size + 1):
        for j in range(width - filter_size + 1):
            roi = img_array[i:i + filter_size, j:j + filter_size, channel]
            blurred_value = np.sum(roi * gaussian_filter)
            blurred_image_array[i + filter_size // 2, j + filter_size // 2, channel] = blurred_value

blurred_image = Image.fromarray(blurred_image_array)

# blurred_image.show()

blurred_image.save('../Reference/data/1.1_blur.jpg') # save to file

print("success")