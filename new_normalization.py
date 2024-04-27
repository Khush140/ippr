#Image Normalization: 24 bit
import cv2
import numpy as np
import matplotlib.pyplot as plt

def normalize_image(image):

    img_normalized = cv2.normalize(image, None, 0, 1.0,
    cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # ret,thresh = cv2.threshold(image,140,255,cv2.THRESH_BINARY)
    # print("Image data after Thresholding:\n", thresh)


    # img_normalized = cv2.normalize(thresh, None, 0, 1.0,
    # cv2.NORM_MINMAX, dtype=cv2.CV_32F)


    return img_normalized



# Read the input RGB image
rgb_image = cv2.imread('24bit_rgb.jpeg')

# Normalize the RGB image
normalized_image = normalize_image(rgb_image)



# Plot the original image, normalized image, and CMY image side by side
plt.figure(figsize=(12, 4))

# Original RGB image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
plt.title('Original RGB Image')
plt.axis('off')

# Normalized image
plt.subplot(1, 3, 2)
plt.imshow(normalized_image)
plt.title('Normalized Image')
plt.axis('off')



plt.show()
