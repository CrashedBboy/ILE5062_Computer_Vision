import cv2
import numpy as np
from matplotlib import pyplot as plt
import os.path as path
import funcs

IMG1 = './data/task1_hybrid/dog.jpg'
IMG2 = './data/task1_hybrid/cat.jpg'

CUT_OFF_RATIO = 0.06

image1_abs_path = path.abspath(path.join(path.dirname(__file__), IMG1))
image2_abs_path = path.abspath(path.join(path.dirname(__file__), IMG2))
if (not path.exists(image1_abs_path)) or (not path.exists(image1_abs_path)):
    print("invalid image path")
    exit()

image1 = cv2.cvtColor(cv2.imread(image1_abs_path), cv2.COLOR_BGR2RGB)
image2 = cv2.cvtColor(cv2.imread(image2_abs_path), cv2.COLOR_BGR2RGB)
if (image1.data == None) or (image2.data == None):
    print("failed to read image")
    exit()

# normalize image's RGB value to range [0, 1]
image1 = image1.astype(np.float32) / 255
image2 = image2.astype(np.float32) / 255

# apply low pass(Gaussian) filter on image1
filtered_image1 = np.zeros(image1.shape, dtype=np.float32)

for ch in range(image1.shape[2]):

    low_pass_filter = funcs.get_gaussian_lp_filter(image1[:,:,ch].shape, CUT_OFF_RATIO)

    filtered_image1[:,:,ch], _ = funcs.frequency_filtering(image1[:,:,ch], low_pass_filter, convert_uint8=False)

# apply high pass(Laplacian) filter on image2
filtered_image2 = np.zeros(image2.shape, dtype=np.float32)

for ch in range(image2.shape[2]):

    low_pass_filter = funcs.get_gaussian_lp_filter(image2[:,:,ch].shape, CUT_OFF_RATIO)

    filtered_image2[:,:,ch], _ = funcs.frequency_filtering(image2[:,:,ch], low_pass_filter, convert_uint8=False)

filtered_image2 = image2 - filtered_image2
filtered_image2[filtered_image2 < 0] = 0

# filtered_image2 = np.ones(filtered_image2.shape, dtype=filtered_image2.dtype) - filtered_image2

length = image1.shape[0]
if image2.shape[0] > image1.shape[0]:
    length = image2.shape[0]

height = image1.shape[1]
if image2.shape[1] > image1.shape[1]:
    height = image2.shape[1]

blended_image = filtered_image1[0:length, 0:height] + filtered_image2[0:length, 0:height]

blended_image = (blended_image - blended_image.min()) / (blended_image.max() - blended_image.min())

plt.subplot(2, 3, 1)
plt.imshow(image1)
plt.title('Original Image 1'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 2)
plt.imshow(image2)
plt.title('Original Image 2'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 3)
plt.imshow(cv2.cvtColor(blended_image, cv2.COLOR_RGB2GRAY), cmap='gray')
plt.title('Blended Image (Grayscale)'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 4)
plt.imshow(filtered_image1)
plt.title('Low-Passed Image 1'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 5)
plt.imshow(filtered_image2)
plt.title('High-Passed Image 2'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 3, 6)
plt.imshow(blended_image)
plt.title('Blended Image'), plt.xticks([]), plt.yticks([])
plt.show()