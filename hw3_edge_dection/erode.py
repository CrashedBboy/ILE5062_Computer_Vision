import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import cv2

import os.path as path
import funcs

# erosion kernel size
KERNEL_SIZE = 5

IMAGE = './data/freedom_gundam_head.jpg'

absolute_image = path.abspath( path.join( path.dirname(__file__), IMAGE ) )
if not path.exists(absolute_image):
    print("image", IMAGE, "is not exist")
    exit()

# read image
image = cv2.imread(absolute_image)

# convert to grayscale image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# convert to binary image (thresholding)
binary_image = np.copy(gray_image)
# binary_image[binary_image < 40] = 0
# binary_image[binary_image >= 40] = 1

# plt.imshow(binary_image, cmap='gray')
# plt.show()

border_width = int((KERNEL_SIZE-1) / 2)

padded_image = np.ones((binary_image.shape[0] + 2*border_width, binary_image.shape[1] + 2*border_width), dtype=np.uint8)

padded_image[border_width:(padded_image.shape[0] - border_width), border_width:(padded_image.shape[1] - border_width)] = binary_image[:,:]

convolved_image = np.zeros(binary_image.shape, dtype=np.uint8)

for r in range(convolved_image.shape[0]):
    for c in range(convolved_image.shape[1]):

        shifted_r = r + border_width
        shifted_c = c + border_width

        min = 255
        keep = True

        for i in range(shifted_r - border_width, shifted_r + border_width + 1):
            if keep:
                for j in range(shifted_c - border_width, shifted_c + border_width + 1):
                    if keep:
                        
                        if (padded_image[i, j] < min):
                            min = padded_image[i, j]

                            if (padded_image[i, j] == 0):
                                keep = False

        convolved_image[r, c] = min

plt.subplot(1,2,1)
plt.imshow(binary_image, cmap='gray')
plt.title('Grayscale Image'), plt.xticks([]), plt.yticks([])
plt.subplot(1,2,2)
plt.imshow(convolved_image, cmap='gray')
plt.title('Eroded with kernel(size=5)'), plt.xticks([]), plt.yticks([])

plt.show()