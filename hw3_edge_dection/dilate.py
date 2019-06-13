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

border_width = int((KERNEL_SIZE-1) / 2)

padded_image = np.zeros((gray_image.shape[0] + 2 * border_width, gray_image.shape[1] + 2 * border_width), dtype=np.uint8)

padded_image[border_width:padded_image.shape[0]-border_width, border_width:padded_image.shape[1]-border_width] = gray_image[:,:]

# top left corner square reflection
for r in range(0, border_width):
    for c in range(0, border_width):
        padded_image[r, c] = padded_image[border_width + (border_width - 1 - r), border_width + (border_width - 1 - c)]

# top center strip reflection
for r in range(0, border_width):
    for c in range(border_width, padded_image.shape[1] - border_width):
        padded_image[r, c] = padded_image[border_width + (border_width - 1 - r), c]

# top right square corner reflection
for r in range(0, border_width):
    for c in range(padded_image.shape[1] - border_width, padded_image.shape[1]):
        padded_image[r, c] = padded_image[border_width + (border_width - 1 - r), (padded_image.shape[1]-1-border_width) - (c - (padded_image.shape[1] - border_width))]

# left center strip reflection
for r in range(border_width, padded_image.shape[0] - border_width):
    for c in range(0, border_width):
        padded_image[r, c] = padded_image[r, border_width + (border_width - 1 - c)]

# left bottom square corner reflection
for r in range(padded_image.shape[0] - border_width, padded_image.shape[0]):
    for c in range(0, border_width):
        padded_image[r, c] = padded_image[(padded_image.shape[0] - border_width - 1) - (r - (padded_image.shape[0] - border_width)), border_width + (border_width - 1 - c)]

# left center strip reflection
for r in range(padded_image.shape[0] - border_width, padded_image.shape[0]):
    for c in range(border_width, padded_image.shape[1] - border_width):
        padded_image[r, c] = padded_image[(padded_image.shape[0] - border_width - 1) - (r - (padded_image.shape[0] - border_width)), c]

# right bottom square corner reflection
for r in range(padded_image.shape[0] - border_width, padded_image.shape[0]):
    for c in range(padded_image.shape[1] - border_width, padded_image.shape[1]):
        padded_image[r, c] = padded_image[(padded_image.shape[0] - border_width - 1) - (r - (padded_image.shape[0] - border_width)), (padded_image.shape[1]-1-border_width) - (c - (padded_image.shape[1] - border_width))]

# right center square strip reflection
for r in range(border_width, padded_image.shape[0] - border_width):
    for c in range(padded_image.shape[1] - border_width, padded_image.shape[1]):
        padded_image[r, c] = padded_image[r, (padded_image.shape[1]-1-border_width) - (c - (padded_image.shape[1] - border_width))]


convolved_image = np.zeros(gray_image.shape, dtype=np.uint8)

# for r in range(convolved_image.shape[0]):
#     for c in range(convolved_image.shape[1]):

        # convolve the image: replace pixel with max value of its neighbor
        # reflect in border