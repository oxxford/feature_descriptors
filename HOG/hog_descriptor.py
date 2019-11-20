from skimage.transform import resize
from numpy.linalg import norm
import cv2
import numpy as np

"""
This is the implementation of HOG descriptor as it was described in original paper
"""

def hog_desc(img, window_size = 8, stride = 4, bin_n = 9):
    """
    This function constructs HOG feature descriptor for an image
    :param img: np array - input image
    :param window_size:
    :param stride:
    :param bin_n:
    :return: vector of features
    """
    height = 128
    width = 64

    proc_img = np.float32(resize(img, (height, width)))
    proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)

    features = []
    for y in range(0, height - window_size, stride):
        for x in range(0, width - window_size, stride):
            wnd = proc_img[y:y + window_size, x:x + window_size]

            # Do edge detection.
            gx = cv2.Sobel(wnd, cv2.CV_32F, 1, 0)
            gy = cv2.Sobel(wnd, cv2.CV_32F, 0, 1)
            mag, ang = cv2.cartToPolar(gx, gy)

            # Bin the angles.
            bin = np.int32(bin_n * ang / (2 * np.pi))
            # the magnitudes are used as weights for the gradient values.
            hist = np.bincount(bin.ravel(), mag.ravel(), bin_n)

            # normalization
            eps = 1e-7
            hist /= hist.sum() + eps
            hist = np.sqrt(hist)
            hist /= norm(hist) + eps
            features.extend(hist)

    # handling irregular length of feature vector
    if len(features) != 3780:
        if len(features) > 3780:
            features = features[:3780]
        if len(features) < 3780:
            features += [0] * (3780 - len(features))

    return np.float32(features)