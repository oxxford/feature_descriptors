from commons.custom_pickle_loader import try_to_load_as_pickled_object_or_None
from commons.svm import SVM
import numpy as np

"""
Here we train five SVMs (different kernels)

For CatBoost training see the hog_catboost.ipynb
or go to: https://colab.research.google.com/drive/1eFhINcjZka73pFHMD1u5c_oeH7nN3Otk
"""

def train_one_svm(svm, descriptors, labels, path_to_save='svm_halfdata_train.dat'):
    """
    This function trains one SVM
    :param svm: object of type SVM()
    :param descriptors: np array of feature vectors (type - float32)
    :param labels: np array of classes
    :param path_to_save: path to save trained model
    :return:
    """
    train_data = np.matrix(descriptors, dtype=np.float32)
    labels = np.array([labels]).transpose()
    svm.train(train_data, labels)
    svm.save(path_to_save)
    return svm

def train_svm():
    """
    This fundtion trains 5 SVMs with different kernels
    :return:
    """
    # loading training feature vectors
    features_train = try_to_load_as_pickled_object_or_None("hog_features_train.pkl")
    nums_train = try_to_load_as_pickled_object_or_None("../main_data/nums_train.pkl")

    kernels = ["linear", "chi", "inter", "rbf", "sigmoid"]

    for kr in kernels:
        path = './models/svm_halfdata_train_' + kr + '.dat'
        svm = SVM(kernel=kr)
        train_one_svm(svm, features_train, nums_train, path_to_save=path)

if __name__ == '__main__':
    train_svm()
