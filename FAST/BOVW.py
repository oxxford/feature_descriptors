import numpy as np
import cv2
import time

from FAST.fast import detect, get_brief_description
from sklearn.cluster import KMeans


def get_feature_list(imgs):
    """
    Go through dataset and collect features from all images
    :param imgs: list of images
    :return: dictionary image_number: list of features
    """
    feature_list = {}

    t1 = time.time()

    for i, img in enumerate(imgs):
        imgray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        imgray = cv2.GaussianBlur(imgray, (3, 3), 1)

        corners = detect(imgray)
        descriptors = get_brief_description(imgray, corners)

        feature_list[i] = descriptors

        if i % 100 == 0:
            print(i, (time.time() - t1) / 60)

    return feature_list


def get_kmeans(feature_list):
    """
    Executer kmeans clustering on collected features to create a codebook
    :param feature_list: dictionary image_number: list of features
    :return: sklearn kmeans object
    """
    sample = []

    for i, item in enumerate(feature_list.values()):
        sample.extend(item)

    kmeans = KMeans(n_clusters=500, random_state=0).fit(sample)

    return kmeans


def get_feature_images(feature_list, kmeans, img_len):
    """
    Combines final features to images
    :param feature_list: dictionary image_number: list of features
    :param kmeans: kmeans object
    :param img_len: number of images
    :return: dictionary image_number: list of final features
    """
    img_features = {}

    t1 = time.time()

    for i in range(img_len):
        descriptors = feature_list[i]

        if len(descriptors) == 0:
            print(i)
            continue

        f = kmeans.predict(descriptors)

        res = np.zeros((500,))

        for j in f:
            res[j] += 1

        img_features[i] = res

        if i % 100 == 0:
            print(i, (time.time() - t1) / 60)

    return img_features
