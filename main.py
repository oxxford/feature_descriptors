from fast import *
from matplotlib import pyplot as plt
import cv2
import numpy as np


def test():
    image = cv2.imread('./download.jpg')
    imgray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    imgray = cv2.GaussianBlur(imgray, (3, 3), 1)

    corners = detect(imgray)
    print(len(corners))

    plt.imshow(imgray, cmap='gray')
    for point in corners:
        plt.scatter(point[0], point[1], s=10, color='r')
    plt.show()


test()
