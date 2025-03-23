from PIL import Image
import numpy as np
from scipy.special import ellip_harm

# Load the image
img = Image.open('../Reference/data/2.4_suppress.jpg')

suppress = np.array(img, dtype=np.uint8)

# Hysteresis thresholding
def hysteresis_thresholding(image, low_threshold, high_threshold):
    high_threshold_value = image.max() * high_threshold
    low_threshold_value = high_threshold_value * low_threshold

    height, width = image.shape
    res = np.zeros((height, width), dtype=np.uint8)

    strong_i, strong_j = np.where(image >= high_threshold_value)
    weak_i, weak_j = np.where((image <= high_threshold_value) & (image >= low_threshold_value))

    res[strong_i, strong_j] = 255

    for i in range(len(weak_i)):
        if (weak_i[i] + 1 < height and image[weak_i[i] + 1, weak_j[i]] == 255) or \
           (weak_i[i] - 1 >= 0 and image[weak_i[i] - 1, weak_j[i]] == 255) or \
           (weak_j[i] + 1 < width and image[weak_i[i], weak_j[i] + 1] == 255) or \
           (weak_j[i] - 1 >= 0 and image[weak_i[i], weak_j[i] - 1] == 255) or \
           (weak_i[i] + 1 < height and weak_j[i] + 1 < width and image[weak_i[i] + 1, weak_j[i] + 1] == 255) or \
           (weak_i[i] - 1 >= 0 and weak_j[i] - 1 >= 0 and image[weak_i[i] - 1, weak_j[i] - 1] == 255):
            res[weak_i[i], weak_j[i]] = 255

    return res

low_threshold_ratio = 0.05
high_threshold_ratio = 0.15

low_threshold_image = hysteresis_thresholding(suppress, low_threshold_ratio, high_threshold_ratio)
high_threshold_image = hysteresis_thresholding(suppress, high_threshold_ratio*2.5 , high_threshold_ratio*3)

final_edge_map_image = hysteresis_thresholding(suppress , low_threshold_ratio , high_threshold_ratio)

# Save the binarized images using low and high thresholds and the final result by Hysteresis thresholding
Image.fromarray(low_threshold_image).save('../Reference/data/2.5_edgemap_low.jpg')
Image.fromarray(high_threshold_image).save('../Reference/data/2.5_edgemap_high.jpg')
Image.fromarray(final_edge_map_image).save('../Reference/data/2.5_edgemap.jpg')


