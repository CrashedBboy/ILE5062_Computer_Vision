import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# draw & show image
def img_show(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

def imgs_compare(image1, image2):
    figure = plt.figure()

    figure.add_subplot(1,2, 1)
    plt.imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))

    figure.add_subplot(1,2, 2)
    plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    
    plt.show(block=True)