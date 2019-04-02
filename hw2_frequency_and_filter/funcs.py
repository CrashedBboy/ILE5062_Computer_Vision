import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# draw & show image
def img_show(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()