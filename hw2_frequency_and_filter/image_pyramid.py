import cv2
import numpy as np
from matplotlib import pyplot as plt
import os.path as path
import math
import funcs

CUT_OFF_RATIO = 0.5
PYRAMID_DEPTH = 3
FILTER = True

IMG = './data/task2_pyramid/grid1.jpg'
OUTPUT_DIR = './reference'

abs_image_path = path.abspath(path.join(path.dirname(__file__), IMG))

if (not path.exists(abs_image_path)):
    print("invalid image path")
    exit()

abs_output_path = path.abspath(path.join(path.dirname(__file__), OUTPUT_DIR))

image_basename = path.basename(abs_image_path)

image = cv2.imread(abs_image_path)

stack = [image]

# OpenCV's solution
'''
for i in range(PYRAMID_DEPTH):
    stack.append(cv2.pyrDown(stack[-1]))

for i, img in enumerate(stack):
    cv2.imwrite(path.join(abs_output_path, 'opencv_level' + str(i) + '_' + image_basename), img)
'''

for level in range(PYRAMID_DEPTH):
    
    img = stack[-1]

    filtered_img = np.zeros(img.shape, dtype=np.uint8)

    # gaussian filtering
    if FILTER:
        for ch in range(img.shape[2]):
            low_pass_filter = funcs.get_gaussian_lp_filter(img[:,:,ch].shape, CUT_OFF_RATIO)
            filtered_img[:,:,ch], _ = funcs.frequency_filtering(img[:,:,ch], low_pass_filter, convert_uint8=True)
    else:
        filtered_img = img

    # sub-sampling
    sampled_img = np.zeros((math.floor(img.shape[0]/2), math.floor(img.shape[1]/2), img.shape[2]), dtype=np.uint8)

    for row in range(sampled_img.shape[0]):
        for column in range(sampled_img.shape[1]):
            for ch in range(sampled_img.shape[2]):
                sampled_img[row, column, ch] = round(filtered_img[row*2, column*2, ch]/4 + 
                    filtered_img[row*2+1, column*2, ch]/4 + 
                    filtered_img[row*2, column*2+1, ch]/4 + 
                    filtered_img[row*2+1, column*2+1, ch]/4)

    stack.append(sampled_img)

for i, img in enumerate(stack):
    cv2.imwrite(path.join(abs_output_path, 'level' + str(i) + '_' + image_basename), img)