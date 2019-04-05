import cv2
import numpy as np
from matplotlib import pyplot as plt
import os.path as path
import math

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

# Compute the 2-dimensional discrete Fourier Transform
frequency = np.fft.fft2(image)

# Shift the zero-frequency component to the center of the spectrum (swap half-space)
shifted_frequency = np.fft.fftshift(frequency)

magnitude_spectrum = np.log(np.abs(shifted_frequency))

# declare low pass filter on frequency domain (element values: [0, 1], pass or not)
low_pass_filter = np.zeros(frequency.shape, dtype=np.bool)

# compute "how low" the frequency can pass
cut_off_frequency = math.ceil(CUT_OFF_RATIO * 0.5 * frequency.shape[0])

if (frequency.shape[0] > frequency.shape[1]):
    cut_off_frequency = math.ceil(CUT_OFF_RATIO * 0.5 * frequency.shape[1])

# write values to filter
# we need to shift the index, [0, 1, 2, 3, 4] --> [-2, -1, 0, 1, 2]; [0, 1, 2, 3] -> [-1.5, -0.5, 0.5, 1.5]
u_offset = (low_pass_filter.shape[0]-1) / 2
v_offset = (low_pass_filter.shape[1]-1) / 2

for u in range(low_pass_filter.shape[0]):
    shifted_u = u - u_offset

    for v in range(low_pass_filter.shape[1]):
        shifted_v = v - v_offset
        radius_from_center = (shifted_u**2 + shifted_v**2)**(0.5)

        if (radius_from_center <= cut_off_frequency):
            low_pass_filter[u, v] = 1

filtered_frequency = shifted_frequency * low_pass_filter

filtered_spectrum = np.log(np.abs(filtered_frequency) + 1) # add 1 to avoid "divided by zero" exception

filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_frequency)).real

# normalize
filtered_image = filtered_image - filtered_image.min()
filtered_image = (filtered_image / filtered_image.max()) * 255
filtered_image = filtered_image.astype(np.uint8)

# print(filtered_image.dtype)
# print(image.dtype), exit()

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

