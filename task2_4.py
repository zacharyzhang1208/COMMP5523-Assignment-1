from PIL import Image
import numpy as np

# Load the image
img1 = Image.open('../Reference/data/2.3_G.jpg')
img2 = Image.open('../Reference/data/2.3_G_x.jpg')
img3 = Image.open('../Reference/data/2.3_G_y.jpg')

gradient_magnitude_uint8 = np.array(img1, dtype=np.uint8)
sobel_x_uint8 = np.array(img2, dtype=np.uint8)
sobel_y_uint8 = np.array(img3, dtype=np.uint8)


# Non-maximum suppression
def non_maximum_suppression(magnitude, sobel_x, sobel_y):
    height, width = magnitude.shape
    suppressed_img = np.zeros((height, width), dtype=np.float32)

    angle = np.arctan2(sobel_y, sobel_x) * (180.0 / np.pi)
    angle[angle < 0] += 180

    for i in range(1, height - 1):
        for j in range(1, width - 1):
            q = 255
            r = 255

            # Angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            # Angle 45
            elif 22.5 <= angle[i, j] < 67.5:
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            # Angle 90
            elif 67.5 <= angle[i, j] < 112.5:
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            # Angle 135
            elif 112.5 <= angle[i, j] < 157.5:
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                suppressed_img[i, j] = magnitude[i, j]
            else:
                suppressed_img[i, j] = 0

    return suppressed_img


suppressed_image = non_maximum_suppression(gradient_magnitude_uint8, sobel_x_uint8, sobel_y_uint8)

# Save the suppressed image
suppressed_image_uint8 = suppressed_image.astype(np.uint8)
Image.fromarray(suppressed_image_uint8).save('../Reference/data/2.4_suppress.jpg')