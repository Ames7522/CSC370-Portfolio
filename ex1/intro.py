'''
Author: Ames Azogue Irigoyen
Last Modified: February 11, 2026
Description: This code takes an image and displays it in the viewer, experiments with different basic image filters using pillow and CV2.
'''
# Getting started with images --------------------------------------------------------------
# Imports Image class from PIL module
from PIL import Image

# Opens image in rbg mode
im = Image.open(r"stars.jpg")           

# Shows the image in image viewer
im.show()   

# Prints the format of the image (in this case JPEG)
print(im.format)

# Prints the size of the image (width, height)
print(im.size)

## Cropping the image ----------------------------------------------------------------------

'''
The following cropping is done by specifying the left, upper, right, and lower pixel coordinate.

The left and upper coordinates are inclusive, while the right and lower coordinates are exclusive, the values can be interpreted as follows:

left: The x-coordinate of the left edge of the crop area.
upper: The y-coordinate of the top edge of the crop area.
right: The x-coordinate of the right edge of the crop area.
lower: The y-coordinate of the bottom edge of the crop area. 
'''
cropped_stars = im.crop((200, 1000, 700, 1500))
cropped_stars.show()

# To crop the image in the center, we can follow the following steps:
width, height = im.size
new_width, new_height = 700, 900 # example dimensions
left = (width - new_width) / 2
top = (height - new_height) / 2
right = (width + new_width) / 2
bottom = (height + new_height) / 2

centered_stars = im.crop((left, top, right, bottom))
centered_stars.show()

# to save the cropped image, we can use the save() method
centered_stars.save("centered_cropped_stars.jpg")
cropped_stars.save("cropped_stars.jpg")

# converting to grayscale
gray = im.convert("L")
gray.show()

# spliting the image into RGB channels
r, g, b = im.split()
r.show()
g.show()
b.show()


# an alternative way to save an image comes in the cv2 module, which is part of the OpenCV library. This method is more efficient for saving images in certain formats, like PNG, and allows for more control over the compression settings. This will be covered in the section titled "OpenCV"

#OpenCV ----------------------------------------------------------------------------------------------
import cv2
import numpy as np

# Convert PIL image to OpenCV format
im_cv = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)

# Convert to grayscale using OpenCV
gray_cv = cv2.cvtColor(im_cv, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale (OpenCV)", gray_cv)

# Blurring and Edge Detection-------------------------------------------------------------------------

# applying Gaussian blur using OpenCV
gBlur = cv2.GaussianBlur(im_cv, (5, 5),0)
cv2.imshow("Gaussian Blur (OpenCV)", gBlur)
cv2.waitKey(0)  # Wait for a key press to close the windows
cv2.destroyAllWindows() # Close all OpenCV windows

#check the effect of a larger kernel size on the blurring of the image
gBlur2 = cv2.GaussianBlur(im_cv, (51, 51),0) 
cv2.imshow("Gaussian Blur (OpenCV)", gBlur2)
cv2.waitKey(0)  # Wait for a key press to close the windows
cv2.destroyAllWindows() # Close all OpenCV windows 

# applying Canny edge detection using OpenCV
edges = cv2.Canny(gray_cv, 100, 200)
cv2.imshow("Edges", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()


