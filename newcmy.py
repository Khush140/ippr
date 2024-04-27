#rgb to cmy: 24 bit
import cv2
import numpy as np
import matplotlib.pyplot as plt

def rgb_to_cmy_pixel_by_pixel(rgb_image):
    """
    Convert an RGB image to a CMY image pixel by pixel using OpenCV.

    Args:
        rgb_image (numpy.ndarray): Input RGB image.

    Returns:
        numpy.ndarray: Converted CMY image.
    """
    height, width, channels = rgb_image.shape
    cmy_image = np.zeros_like(rgb_image)

    for y in range(height):
        for x in range(width):
            r, g, b = rgb_image[y, x]
            c = 255 - r
            m = 255 - g
            y_val = 255 - b
            cmy_image[y, x] = [c, m, y_val]

    return cmy_image

# Load the RGB image
rgb_image = cv2.imread('24bit_rgb.jpeg')

# Convert BGR to RGB
rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

# Convert RGB to CMY pixel by pixel
cmy_image = rgb_to_cmy_pixel_by_pixel(rgb_image)

# Display the images
plt.subplot(1, 2, 1)
plt.imshow(rgb_image)
plt.title('Original RGB Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cmy_image)
plt.title('CMY Image')
plt.axis('off')

plt.show()
