import cv2
import matplotlib.pyplot as plt
import numpy as np

def show_img(image, ticks=False):

    plt.imshow( cv2.cvtColor(image, cv2.COLOR_BGR2RGB) )

    if not ticks:
        plt.xticks([])
        plt.yticks([])

    plt.show()