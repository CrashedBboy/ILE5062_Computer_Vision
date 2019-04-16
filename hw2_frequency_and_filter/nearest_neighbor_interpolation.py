import cv2
import numpy as np
import matplotlib.pyplot as plt
import os.path as path
import funcs
import math

SCALE = 8

IMG = './data/task2_pyramid/small_cat.jpg'

OUTPUT_DIR = './reference'

abs_image_path = path.abspath( path.join( path.dirname(__file__), IMG) )

if not path.exists(abs_image_path):
    print("image not found:", absolute_img_path)
    exit()

abs_output_path = path.abspath(path.join(path.dirname(__file__), OUTPUT_DIR))

image_basename = path.basename(abs_image_path)

image = cv2.imread(abs_image_path)

if image.data == None:
    print("invalid image:", abs_image_path)
    exit()

scaled_width = math.floor(image.shape[0] * SCALE)
scaled_height = math.floor(image.shape[1] * SCALE)

scaled_image = np.zeros((scaled_width, scaled_height, image.shape[2]), image.dtype)

for row in range(scaled_image.shape[0]):
    for column in range(scaled_image.shape[1]):
        for ch in range(scaled_image.shape[2]):
            nearest_x = round( ((row+1)/scaled_image.shape[0]) * (image.shape[0]-1) )
            nearest_y = round( ((column+1)/scaled_image.shape[1]) * (image.shape[1]-1) )
            
            scaled_image[row, column, ch] = image[nearest_x, nearest_y, ch]

cv2.imwrite(path.join(abs_output_path, 'scaled_nn_' + str(SCALE) + '_' + image_basename), scaled_image)