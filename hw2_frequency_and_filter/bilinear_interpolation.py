import cv2
import numpy as np
import matplotlib.pyplot as plt
import os.path as path
import funcs
import math

SCALE = 4

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
            x = ((row+1)/scaled_image.shape[0]) * (image.shape[0]-1)
            lower_x = math.floor(x)
            upper_x = math.ceil(x)
            offset_x = x - lower_x

            y = ((column+1)/scaled_image.shape[1]) * (image.shape[1]-1)
            lower_y = math.floor(y)
            upper_y = math.ceil(y)
            offset_y = y - lower_y

            upper_interpolated = image[lower_x, upper_y, ch] * (1-offset_x) + image[upper_x, upper_y, ch] * (offset_x)
            lower_interpolated = image[lower_x, lower_y, ch] * (1-offset_x) + image[upper_x, lower_y, ch] * (offset_x)

            interpolated = lower_interpolated * (1-offset_y) + upper_interpolated * offset_y
            
            scaled_image[row, column, ch] = round(interpolated)

cv2.imwrite(path.join(abs_output_path, 'bl_scaled_' + str(SCALE) + '_' + image_basename), scaled_image)