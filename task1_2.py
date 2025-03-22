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

# define 3*3 sharpening filter and parameter 'a' according to
# (1) Detailed = Original - Blurred
# (2) Sharpened = Original + a * Detailed
# Thus, Sharpened = (1 + a) * Original - a * Blurred
# Here we choose a relatively large a
a = 10
original = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])
sharpening_filter = (1 + a) * original - a * gaussian_filter
# print(sharpening_filter)

# Create an output array to store the sharpened image
sharpened_image_array = np.zeros_like(img_array)

height, width, channels = img_array.shape
# print(img_array.shape)

#This method actually does not consider handling the padding. But I think that's enough for task 2 as well as task 1.
for channel in range(channels):
    for row in range(height - filter_size + 1):
        for col in range(width - filter_size + 1):
            roi = img_array[row:row + filter_size, col:col + filter_size, channel]
            sharpened_value = np.sum(roi * sharpening_filter)
            sharpened_image_array[row + filter_size // 2, col + filter_size // 2, channel] = sharpened_value

sharpened_image = Image.fromarray(sharpened_image_array)
# sharpened_image.show()

sharpened_image.save('../Reference/data/1.2_sharpened.jpg') # save to file

print("success")