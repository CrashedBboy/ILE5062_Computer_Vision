import cv2
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt
import os.path as path
import math

IMAGE = 'data/chessboard.jpg'

G_KERNEL_SIZE = 3
G_SIGMA = 3

# load image
image_path = path.abspath( path.join( path.dirname(__file__), IMAGE ) )

image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# do gaussian blurring to remove noise
if (G_KERNEL_SIZE % 2) == 0:
    print("kernel width/height of gaussian filter has to be even number, increase 1 automatically")
    G_KERNEL_SIZE += 1

x, y = np.mgrid[ (-0.5)*(G_KERNEL_SIZE-1):(0.5)*(G_KERNEL_SIZE-1)+1, (-0.5)*(G_KERNEL_SIZE-1):(0.5)*(G_KERNEL_SIZE-1)+1 ]

gaussian_kernel = np.exp((-1) * (x**2 + y**2)/(2*(G_SIGMA**2))) / (2*np.pi*(G_SIGMA**2))

# normalize kernel
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

print("Kernel size:", G_KERNEL_SIZE, ", sigma value:", G_SIGMA, ", Gaussian kernel: \n", gaussian_kernel)

blurred_image = signal.convolve2d(gray_image, gaussian_kernel, boundary='symm', mode='same')

# compute x, y gradiant (using sobel)

sobel_x = np.array([
    -1, 0, 1,
    -2, 0, 2,
    -1, 0, 1
])

sobel_x = sobel_x.reshape((3,3))

sobel_y = np.array([
    -1, -2, -1,
    0, 0, 0,
    1, 2, 1
])

sobel_y = sobel_y.reshape((3,3))

gradient_x = signal.convolve2d(blurred_image, sobel_x, boundary='symm', mode='same')
gradient_y = signal.convolve2d(blurred_image, sobel_y, boundary='symm', mode='same')

gradient_xy = gradient_x * gradient_y

# using gradiant to do harris dection

derterminant = gradient_x**2 * gradient_y**2 - gradient_xy**2
trace = gradient_x**2 + gradient_y**2 + (1e-12) # add a small number to avoid "divide by 0"

corner_response = np.abs(derterminant / trace)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(1, 2, 2)
plt.imshow(corner_response, cmap='gray')
plt.title('Corner Response'), plt.xticks([]), plt.yticks([])

plt.show()

# thresholding? non-maximize suppression