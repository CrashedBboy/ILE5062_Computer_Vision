import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math

# draw & show image
def img_show(image):
    plt.imshow(image)
    plt.show()

def imgs_compare(image1, image2):
    figure = plt.figure()

    figure.add_subplot(1,2, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

    figure.add_subplot(1,2, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    
    plt.show(block=True)

# shape: tuple, cut_off_ratio: float
def get_ideal_lp_filter(shape, cut_off_ratio):
    # declare low pass filter on frequency domain (element values: [0, 1], pass or not)
    low_pass_filter = np.zeros(shape, dtype=np.bool)

    # compute "how low" the frequency can pass
    cut_off_frequency = math.ceil(cut_off_ratio * 0.5 * shape[0])

    if (shape[0] > shape[1]):
        cut_off_frequency = math.ceil(cut_off_ratio * 0.5 * shape[1])

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

    return low_pass_filter

# image: grayscale image or image with only one channel(shape=M*N*1), filter: matrix size must equal to image's 
def frequency_filtering(image, image_filter, convert_uint8 = True):

    # Compute the 2-dimensional discrete Fourier Transform
    frequency = np.fft.fft2(image)

    # Shift the zero-frequency component to the center of the spectrum (swap half-space)
    shifted_frequency = np.fft.fftshift(frequency)

    magnitude_spectrum = np.log(np.abs(shifted_frequency))

    filtered_frequency = shifted_frequency * image_filter

    filtered_spectrum = np.log(np.abs(filtered_frequency) + 1) # add 1 to avoid "divided by zero" exception

    filtered_image = np.fft.ifft2(np.fft.ifftshift(filtered_frequency)).real

    filtered_image[filtered_image < 0] = 0

    if convert_uint8:
        filtered_image = filtered_image.astype(np.uint8)

    return filtered_image, filtered_spectrum

# get frequency domain(shifted) of an image
def get_image_frequency(image):
    # Compute the 2-dimensional discrete Fourier Transform
    frequency = np.fft.fft2(image)

    # Shift the zero-frequency component to the center of the spectrum (swap half-space)
    shifted_frequency = np.fft.fftshift(frequency)

    magnitude_spectrum = np.log(np.abs(shifted_frequency))

    return magnitude_spectrum

# shape: tuple, cut_off_ratio: float
def get_gaussian_lp_filter(shape, cut_off_ratio):
    # declare low pass filter on frequency domain (element values: [0, 1], pass or not)
    gaussian_filter = np.zeros(shape, dtype=np.float64)

    # compute "how low" the frequency can pass (radius of the circle)
    cut_off_frequency = math.ceil(cut_off_ratio * 0.5 * shape[0])

    if (shape[0] > shape[1]):
        cut_off_frequency = math.ceil(cut_off_ratio * 0.5 * shape[1])
    
    # write values to filter
    # we need to shift the index, [0, 1, 2, 3, 4] --> [-2, -1, 0, 1, 2]; [0, 1, 2, 3] -> [-1.5, -0.5, 0.5, 1.5]
    u_offset = (gaussian_filter.shape[0]-1) / 2
    v_offset = (gaussian_filter.shape[1]-1) / 2

    for u in range(gaussian_filter.shape[0]):
        shifted_u = u - u_offset

        for v in range(gaussian_filter.shape[1]):
            shifted_v = v - v_offset
            radius_from_center = (shifted_u**2 + shifted_v**2)**(0.5)

            gaussian_filter[u, v] = math.exp( (-1) * radius_from_center**2 / (2 * cut_off_frequency**2))

    return gaussian_filter