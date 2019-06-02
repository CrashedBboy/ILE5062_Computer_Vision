import cv2
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt
import os.path as path
import math
import funcs

# parameters of gaussian
G_KERNEL_SIZE = 3
G_SIGMA = 1

# parameters of double threshold
STRONG_THRESHOLD = 0.3
WEAK_THRESHOLD = 0.1
STRONG_VALUE = 255
WEAK_VALUE = 50

IMAGE = './data/freedom_gundam_head.jpg'

image_path = path.abspath( path.join( path.dirname(__file__), IMAGE) )

image = cv2.imread(image_path)

if image is None:
    print("cannot open image", IMAGE)
    exit()

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# step 1. do Gaussian blurring

if (G_KERNEL_SIZE % 2) == 0:
    print("kernel width/height of gaussian filter has to be even number, increase 1 automatically")
    G_KERNEL_SIZE += 1

x, y = np.mgrid[ (-0.5)*(G_KERNEL_SIZE-1):(0.5)*(G_KERNEL_SIZE-1)+1, (-0.5)*(G_KERNEL_SIZE-1):(0.5)*(G_KERNEL_SIZE-1)+1 ]

gaussian_kernel = np.exp((-1) * (x**2 + y**2)/(2*(G_SIGMA**2))) / (2*np.pi*(G_SIGMA**2))

# normalize
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

# print("Kernel size:", G_KERNEL_SIZE, ", sigma value:", G_SIGMA, ", Gaussian kernel: \n", gaussian_kernel)

blurred_image = signal.convolve2d(gray_image, gaussian_kernel, boundary='symm', mode='same')

# step 2. use sobel operator to find edge (gradient)

sobel_x = np.array(
    [-1, 0, 1,
    -2, 0, 2,
    -1, 0, 1]
    )

sobel_x = sobel_x.reshape((3,3))

sobel_y = np.array(
    [-1, -2, -1,
    0, 0, 0,
    1, 2, 1]
    )

sobel_y = sobel_y.reshape((3,3))

gradient_x = signal.convolve2d(blurred_image, sobel_x, boundary='symm', mode='same')
gradient_y = signal.convolve2d(blurred_image, sobel_y, boundary='symm', mode='same')

gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)

gradient_orientation = np.arctan2(gradient_y, gradient_x)

# step 3. do non-maximum suppression

thin_edges = np.copy(gradient_magnitude)

for r in range(1, thin_edges.shape[0] - 1):
    for c in range(1, thin_edges.shape[1] - 1):

        radian = gradient_orientation[r,c]

        if radian < 0:
            radian += math.pi

        degree = radian * (180 / math.pi)

        n1 = 0
        n2 = 0

        if (degree >= 0 and degree < 22.5) or (degree >= 157.5 and degree <= 180):
            # compare with left and right neighbor

            n1 = thin_edges[r, c-1]
            n2 = thin_edges[r, c+1]

        elif (degree >= 22.5 and degree < 67.5):
            # compare with upper right and lower left neighbor

            n1 = thin_edges[r-1, c+1]
            n2 = thin_edges[r+1, c-1]

        elif (degree >= 67.5 and degree < 112.5):
            # compare with upper and lower neighbor

            n1 = thin_edges[r-1, c]
            n2 = thin_edges[r+1, c]

        elif (degree >= 112.5 and degree < 157.5):
            # compare with upper left and lower right

            n1 = thin_edges[r-1, c-1]
            n2 = thin_edges[r+1, c+1]
        
        center = thin_edges[r,c]

        if (center < n1) or (center < n2):
            thin_edges[r,c] = 0

# step 4. do double thresholing

thresholded = np.copy(thin_edges)

strong_threshold = thresholded.max() * STRONG_THRESHOLD
weak_threshold = thresholded.max() * WEAK_THRESHOLD

thresholded[thresholded < weak_threshold] = 0
thresholded[(thresholded < strong_threshold) & (thresholded >= weak_threshold)] = WEAK_VALUE
thresholded[thresholded >= strong_threshold] = STRONG_VALUE

thresholded = thresholded.astype(np.uint8)

plt.subplot(1, 3, 1)
plt.imshow(gradient_magnitude, cmap='gray')
plt.title('Gradient Magnitude'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 2)
plt.imshow(thin_edges, cmap='gray')
plt.title('non-maximum suppression'), plt.xticks([]), plt.yticks([])
plt.subplot(1, 3, 3)
plt.imshow(thresholded, cmap='gray')
plt.title('double thresholded'), plt.xticks([]), plt.yticks([])
plt.show()