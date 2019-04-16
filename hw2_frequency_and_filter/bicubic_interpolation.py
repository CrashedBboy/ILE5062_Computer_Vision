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
            x = ((row+1)/scaled_image.shape[0]) * (image.shape[0]-1)
            
            # find 2 right neighbor and 2 left neighbor
            neighbor_x = np.array([math.floor(x)-1, math.floor(x), math.ceil(x), math.ceil(x)+1], dtype=np.int32)

            if neighbor_x[0] < 0:
                neighbor_x[0] = 0
            
            if neighbor_x[3] > (image.shape[0]-1):
                neighbor_x[3] = (image.shape[0]-1)
            
            shifted_x = x - neighbor_x[1]

            y = ((column+1)/scaled_image.shape[1]) * (image.shape[1]-1)
            
            # find 2 right neighbor and 2 left neighbor
            neighbor_y = np.array([math.floor(y)-1, math.floor(y), math.ceil(y), math.ceil(y)+1], dtype=np.int32)

            if neighbor_y[0] < 0:
                neighbor_y[0] = 0
            
            if neighbor_y[3] > (image.shape[1]-1):
                neighbor_y[3] = (image.shape[1]-1)
            
            shifted_y = y - neighbor_y[1]

            x_interpolated = []

            # interpolation in x(row) direction
            b = np.array([shifted_x**3, shifted_x**2, shifted_x**1, 1]).reshape((1,4))
            B_inverse = np.array([
                [-0.167, 0.5, -0.5, 0.167],
                [0.5, -1, 0.5, 0],
                [-0.333, -0.5, 1, -0.167],
                [0, 1, 0, 0]
                ])
            
            for ny in neighbor_y:
                
                F = np.array([image[neighbor_x[0],ny,ch], image[neighbor_x[1],ny,ch], image[neighbor_x[2],ny,ch], image[neighbor_x[3],ny,ch]]).reshape((4,1))

                x_interpolated.append(np.matmul(np.matmul(b, B_inverse), F)[0])

            # interpolation in y(column) direction
            b = np.array([shifted_y**3, shifted_y**2, shifted_y**1, 1]).reshape((1,4))

            F = np.array(x_interpolated).reshape((4,1))

            interpolated = np.matmul(np.matmul(b, B_inverse), F)[0]

            if (interpolated[0] > 255):
                interpolated[0] = 255
            
            if (interpolated[0] < 0):
                interpolated[0] = 0
            
            scaled_image[row, column, ch] = round(interpolated[0])

cv2.imwrite(path.join(abs_output_path, 'scaled_bc_' + str(SCALE) + '_' + image_basename), scaled_image)