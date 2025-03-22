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

# Create an output array to store the blurred image
blurred_image_array = np.zeros_like(img_array)

height, width, channels = img_array.shape
# print(img_array.shape)

#This method actually does not consider handling the padding. But I think that's enough for task 1.
for channel in range(channels):
    for row in range(height - filter_size + 1):
        for col in range(width - filter_size + 1):
            roi = img_array[row:row + filter_size, col:col + filter_size, channel]
            blurred_value = np.sum(roi * gaussian_filter)
            blurred_image_array[row + filter_size // 2, col + filter_size // 2, channel] = blurred_value

blurred_image = Image.fromarray(blurred_image_array)

# blurred_image.show()

blurred_image.save('../Reference/data/1.1_blur.jpg') # save to file

print("success")