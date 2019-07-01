import cv2
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt
import os.path as path
import math


IMAGE = 'data/triangles.jpg'

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

# using gradiant to do harris dection

# thresholding