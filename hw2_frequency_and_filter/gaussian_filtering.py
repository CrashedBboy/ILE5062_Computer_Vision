import cv2
import numpy as np
from os import path
import funcs

IMG = './data/task1and2_hybrid_pyramid/cat.jpg'

# parameters of gaussian
G_KERNEL_SIZE = 3
G_SIGMA = 1

absolute_img_path = path.abspath( path.join( path.dirname(__file__), IMG) )

if not path.exists(absolute_img_path):
    print("image not found:", absolute_img_path)
    exit()

image = cv2.imread(absolute_img_path)

if image.data == None:
    print("invalid image:", absolute_img_path)
    exit()

# funcs.img_show(image)

if (G_KERNEL_SIZE % 2) == 0:
    print("kernel width/height of gaussian filter has to be even number, increase 1 automatically")
    G_KERNEL_SIZE += 1

x, y = np.mgrid[ (-0.5)*(G_KERNEL_SIZE-1):(0.5)*(G_KERNEL_SIZE-1)+1, (-0.5)*(G_KERNEL_SIZE-1):(0.5)*(G_KERNEL_SIZE-1)+1 ]

gaussian_kernel = np.exp((-1) * (x**2 + y**2)/(2*(G_SIGMA**2))) / (2*np.pi*(G_SIGMA**2))

# normalize
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

print("Kernel size:", G_KERNEL_SIZE, ", sigma value:", G_SIGMA, ", Gaussian kernel: \n", gaussian_kernel)