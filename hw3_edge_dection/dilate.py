import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import cv2

import os.path as path
import funcs

# dilation kernel size
KERNEL_SIZE = 3

IMAGE = './data/freedom_gundam_edge.jpg'

absolute_image = path.abspath( path.join( path.dirname(__file__), IMAGE ) )
if not path.exists(absolute_image):
    print("image", IMAGE, "is not exist")
    exit()

# read image
image = cv2.imread(absolute_image)

# convert to grayscale image
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

convolved_image = np.zeros(gray_image.shape, dtype=np.uint8)

for r in range(convolved_image.shape[0]):
    for c in range(convolved_image.shape[1]):

        # convolve the image: replace pixel with max value of its neighbor
        # reflect in border