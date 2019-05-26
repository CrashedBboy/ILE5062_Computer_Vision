import cv2
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt
import os.path as path
import funcs

IMAGE = 'data/freedom_gundam_head.jpg'

THRESHOLD = 0.2

image_abs_path = path.abspath(path.join( path.dirname(__file__), IMAGE))

if not path.exists(image_abs_path):
    print("Test image '" + IMAGE + "' does not exist, exit.")
    exit()

# read image
image = cv2.imread(image_abs_path)

grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# funcs.show_img(image)

# define sobel operator kernel
sobel_x = np.array(
    [-1, 0, 1,
    -2, 0, 2,
    -1, 0, 1],
    np.int8
    )

sobel_x = sobel_x.reshape((3,3))

sobel_y = np.array(
    [-1, -2, -1,
    0, 0, 0,
    1, 2, 1],
    np.int8
    )

sobel_y = sobel_y.reshape((3,3))

gradient_x = signal.convolve2d(grayscale_image, sobel_x, boundary='symm', mode='same')
gradient_y = signal.convolve2d(grayscale_image, sobel_y, boundary='symm', mode='same')

gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

gradient_orientation = np.arctan2(gradient_y, gradient_x)

threshold_mask = np.copy(gradient_magnitude)

threshold = threshold_mask.min() + (threshold_mask.max() - threshold_mask.min()) * THRESHOLD

threshold_mask[threshold_mask >= threshold] = 1
threshold_mask[threshold_mask < threshold] = 0

threshold_mask.astype(np.bool)

thresholded_magnitude = gradient_magnitude * threshold_mask
thresholded_orientation = np.arctan2(gradient_y * threshold_mask, gradient_x * threshold_mask)

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3)
plt.imshow(thresholded_magnitude, cmap='gray')
plt.title('Gradient Magnitude(thresholded)'), plt.xticks([]), plt.yticks([])
plt.show()

plt.imshow(thresholded_orientation, cmap='jet')
plt.title('edge orientation'), plt.xticks([]), plt.yticks([])
plt.colorbar()
plt.show()