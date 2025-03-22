from PIL import Image

# Load the color image
image = Image.open('../Reference/data/road.jpg')

# Convert the color image to grayscale
gray_image = image.convert('L') #L means Luminance
gray_image.show()

# Save the grayscale image
gray_image.save('../Reference/data/2.1_gray.jpg')

print("The grayscale image has been saved as data/2.1_gray.jpg")