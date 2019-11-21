from commons.custom_pickle_loader import try_to_load_as_pickled_object_or_None
from commons.svm import SVM
from FAST.fast import *
from FAST.BOVW import *
from matplotlib import pyplot as plt
import cv2
import numpy as np
import time
from sklearn.metrics import accuracy_score

"""
Training code here is just for the demonstration. 
Actual training process was done in https://drive.google.com/open?id=11UIlDa06doJyN6Ynz_eaKAUH9bc3w4QM
You can check it in FAST training.ipynb
"""


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
    
    # del nums[140]
    # del nums[638]
    # del nums[816]

    model = SVM(kernel="linear")

    vals = np.array(list(img_features.values()), dtype=np.float32)
    res = np.array(nums).reshape(-1, 1)

    model.train(vals, res)
    prediction = model.model.predict(vals)[1]

    print(accuracy_score(prediction, res))


test()
