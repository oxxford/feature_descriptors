from commons.custom_pickle_loader import try_to_load_as_pickled_object_or_None
from commons.svm import SVM
from fast import *
from matplotlib import pyplot as plt
import cv2
import numpy as np
import time
from sklearn.metrics import accuracy_score
from BOVW import get_feature_list, get_kmeans


def get_brief_description(imgray, features, patch_size=48, size=128):
    result = []

    for feature in features:
        if feature[0] < patch_size / 2 or feature[0] > imgray.shape[0] - patch_size / 2 or \
                feature[1] < patch_size / 2 or feature[1] > imgray.shape[1] - patch_size / 2:
            continue

        descriptor = []

        for i in range(size):
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


def get_feature_images(feature_list, kmeans, img_len):
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


def test():
    """
    image = cv2.imread('./download.jpg')
    imgray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    imgray = cv2.GaussianBlur(imgray, (3, 3), 1)

    corners = detect(imgray)
    descriptors = get_brief_description(imgray, corners)
    print(descriptors)

    plt.imshow(imgray, cmap='gray')
    for point in corners:
        plt.scatter(point[0], point[1], s=10, color='r')
    plt.show()
    """
    imgs = try_to_load_as_pickled_object_or_None("imgs.pkl")
    nums = try_to_load_as_pickled_object_or_None("nums.pkl")

    feature_list = get_feature_list(imgs)
    kmeans = get_kmeans(feature_list)

    img_features = get_feature_images(feature_list, kmeans, len(imgs))

    """
    del nums[140]
    del nums[638]
    del nums[816]
    """

    model = SVM(kernel="linear")

    vals = np.array(list(img_features.values()), dtype=np.float32)
    res = np.array(nums).reshape(-1, 1)

    model.train(vals[0:1500], res[0:1500])
    prediction = model.model.predict(vals[1500:])[1]

    print(accuracy_score(prediction, res[1500:]))


test()
