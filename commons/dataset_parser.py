import glob
import os
from skimage.io import imread

"""
This function implements 
"""


def get_data(path="./dog_dataset"):
    """
    :param path: path to dataset folder
    :return: imgs - list of images
             nums - list of corresponding numerical breed-values
             dict{numerical breed: name of breed}
    """
    labels = os.listdir(path)
    labels.remove(".DS_Store")
    breeds = {}
    nums = []
    imgs = []
    for label in labels: # TODO: remove bounding
        label_path = path + "/" + label
        files = os.listdir(label_path)
        name = label[(label.find("-")+1):]
        num = int(label[1:label.find("-")])
        breeds[num] = name
        for file in files:
            file_path = label_path + "/" + file
            img = imread(file_path)
            imgs.append(img)
            nums.append(num)
    return imgs, nums, breeds