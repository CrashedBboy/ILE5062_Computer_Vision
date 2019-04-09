import cv2
import numpy as np
from matplotlib import pyplot as plt
import os.path as path
import math
import funcs

IMG = './data/task1and2_hybrid_pyramid/cat.jpg'

CUT_OFF_RATIO = 0.2

absolute_img_path = path.abspath( path.join( path.dirname(__file__), IMG) )

if not path.exists(absolute_img_path):
    print("image not found:", absolute_img_path)
    exit()


image = cv2.imread(absolute_img_path, cv2.IMREAD_GRAYSCALE)

if image.data == None:
    print("invalid image:", absolute_img_path)
    exit()
 
magnitude_spectrum = funcs.get_image_frequency(image) # not necessory, just for showing

low_pass_filter = funcs.get_ideal_lp_filter(image.shape, CUT_OFF_RATIO)

filtered_image, filtered_spectrum = funcs.frequency_filtering(image, low_pass_filter)

plt.subplot(2, 2, 1),
plt.imshow(image, cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2)
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3),
plt.imshow(filtered_image, cmap = 'gray')
plt.title('Filter Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4)
plt.imshow(filtered_spectrum, cmap = 'gray')
plt.title('Filtered Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
