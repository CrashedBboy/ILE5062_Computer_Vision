import cv2
import numpy as np
from os import path
import math
import funcs

IMG = './data/task1_hybrid/cat.jpg'

# parameters of gaussian
G_KERNEL_SIZE = 9
G_SIGMA = 10

absolute_img_path = path.abspath( path.join( path.dirname(__file__), IMG) )

if not path.exists(absolute_img_path):
    print("image not found:", absolute_img_path)
    exit()

image = cv2.imread(absolute_img_path)

if image.data == None:
    print("invalid image:", absolute_img_path)
    exit()

if (G_KERNEL_SIZE % 2) == 0:
    print("kernel width/height of gaussian filter has to be even number, increase 1 automatically")
    G_KERNEL_SIZE += 1

x, y = np.mgrid[ (-0.5)*(G_KERNEL_SIZE-1):(0.5)*(G_KERNEL_SIZE-1)+1, (-0.5)*(G_KERNEL_SIZE-1):(0.5)*(G_KERNEL_SIZE-1)+1 ]

gaussian_kernel = np.exp((-1) * (x**2 + y**2)/(2*(G_SIGMA**2))) / (2*np.pi*(G_SIGMA**2))

# normalize
gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

print("Kernel size:", G_KERNEL_SIZE, ", sigma value:", G_SIGMA, ", Gaussian kernel: \n", gaussian_kernel)

convolved_image = np.zeros(image.shape, dtype=np.uint8)

for i in range(image.shape[0]):
    for j in range(image.shape[1]):
        for co in range(image.shape[2]):

            sum = 0
            
            # convolve here
            for u, _ in enumerate(gaussian_kernel):
                for v, weight in enumerate(gaussian_kernel[u]):
                    # shift u v index (eg. [0, 1, 2] --> [-1, 0, 1]; [0, 1, 2, 3, 4] --> [-2, -1, 0, 1, 2])
                    shifted_u = u - (G_KERNEL_SIZE-1)/2
                    shifted_v = v - (G_KERNEL_SIZE-1)/2

                    image_index_i = i - shifted_u
                    image_index_j = j - shifted_v

                    # reflect across border
                    image_index_i = math.fabs(image_index_i)
                    image_index_i -= math.floor(image_index_i/(image.shape[0]-1))*(image_index_i % (image.shape[0]-1))
                    image_index_i = int(image_index_i)
                    image_index_j = math.fabs(image_index_j)
                    image_index_j -= math.floor(image_index_j/(image.shape[1]-1))*(image_index_j % (image.shape[1]-1))
                    image_index_j = int(image_index_j)

                    # print("i", i, "j", j, "u", u, "v", v, "shifted_u", shifted_u, "shifted_v", shifted_v, "image_index_i", image_index_i, "image_index_j", image_index_j)

                    sum += gaussian_kernel[u, v] * image[image_index_i, image_index_j, co]

            convolved_image[i, j, co] = int(sum)

funcs.imgs_compare(image, convolved_image)