import cv2
import numpy as np
import scipy.signal as signal
from matplotlib import pyplot as plt
import os.path as path
import funcs

# parameters of gaussian
G_KERNEL_SIZE = 5
G_SIGMA = 1.4

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

print("Kernel size:", G_KERNEL_SIZE, ", sigma value:", G_SIGMA, ", Gaussian kernel: \n", gaussian_kernel)

blurred_image = signal.convolve2d(gray_image, gaussian_kernel, boundary='symm', mode='same')

# step 2. use sobel operator to find edge (gradient)