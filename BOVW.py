import numpy as np
import cv2
import time

from fast import detect
from main import get_brief_description
from sklearn.cluster import KMeans


def get_feature_list(imgs):
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
    sample = []

    for i, item in enumerate(feature_list.values()):
        sample.extend(item)

    kmeans = KMeans(n_clusters=500, random_state=0).fit(sample)

    return kmeans