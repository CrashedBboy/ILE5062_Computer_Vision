import cv2
import numpy as np
from matplotlib import pyplot as plt
import os.path as path
import funcs

IMG = './data/task1_hybrid/cat.jpg'

CUT_OFF_RATIO = 0.03

absolute_img_path = path.abspath( path.join( path.dirname(__file__), IMG) )

if not path.exists(absolute_img_path):
    print("image not found:", absolute_img_path)
    exit()

image = cv2.cvtColor(cv2.imread(absolute_img_path), cv2.COLOR_BGR2RGB)
image = image.astype(np.float32) / 255

if image.data == None:
    print("invalid image:", absolute_img_path)
    exit()

magnitude_spectrum = funcs.get_image_frequency(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY )) # not necessory, just for showing

filtered_image = np.zeros(image.shape, dtype=np.float32)

for ch in range(image.shape[2]):

    low_pass_filter = funcs.get_gaussian_lp_filter(image[:,:,ch].shape, CUT_OFF_RATIO)

    filtered_image[:,:,ch], _ = funcs.frequency_filtering(image[:,:,ch], low_pass_filter, False)

filtered_image = image - filtered_image
filtered_image[filtered_image < 0] = 0

filtered_spectrum = funcs.get_image_frequency(cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY ))
 
plt.subplot(2, 2, 1),
plt.imshow(image)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2)
plt.imshow(magnitude_spectrum, cmap = 'gray')
plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3),
plt.imshow(filtered_image)
plt.title('Filter Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4)
plt.imshow(filtered_spectrum, cmap = 'gray')
plt.title('Filtered Spectrum'), plt.xticks([]), plt.yticks([])
plt.show()
