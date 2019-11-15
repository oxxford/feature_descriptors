from fast import *
from matplotlib import pyplot as plt
import cv2
import numpy as np
from skimage.feature import (match_descriptors, corner_peaks, corner_harris,
                             plot_matches, BRIEF)


def get_brief_description(imgray, features, patch_size=48, size=128):
    result = []

    for feature in features:
        if feature[0] < patch_size / 2 or feature[0] > imgray.shape[0] - patch_size / 2 or \
                feature[1] < patch_size / 2 or feature[1] > imgray.shape[1] - patch_size / 2:
            continue

        descriptor = []

        for i in range(128):
            first_x = feature[0] + np.random.randint(- patch_size // 2, patch_size // 2)
            first_y = feature[1] + np.random.randint(- patch_size // 2, patch_size // 2)

            second_x = feature[0] + np.random.randint(- patch_size // 2, patch_size // 2)
            second_y = feature[1] + np.random.randint(- patch_size // 2, patch_size // 2)

            if imgray[first_x][first_y] < imgray[second_x][second_y]:
                descriptor.append(0)
            else:
                descriptor.append(1)

        result.append(descriptor)

    return result


def test():
    image = cv2.imread('./download.jpg')
    imgray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    imgray = cv2.GaussianBlur(imgray, (3, 3), 1)

    corners = detect(imgray)
    descriptors = get_brief_description(imgray, corners)
    print(descriptors)

    """
    keypoints1 = corner_peaks(corner_harris(imgray), min_distance=5)
    print(keypoints1)
    extractor = BRIEF()

    extractor.extract(imgray, keypoints1)
    descriptors1 = extractor.descriptors
    print(descriptors1)
    """

    plt.imshow(imgray, cmap='gray')
    for point in corners:
        plt.scatter(point[0], point[1], s=10, color='r')
    plt.show()


test()
