from PIL import Image
import numpy as np

img = Image.open('../Reference/data./library.jpg') # read image file
img.show() # show picture
arr = np.asarray(img, dtype=np.float64) # convert to np.ndarray

# process the array here, e.g.
arr = arr/2

arr = arr.astype(np.uint8) # make sure dtype is uint8
img = Image.fromarray(arr) # convert back to PIL.image object
# img.save('output.jpg')          # save to file