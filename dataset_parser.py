import glob
import os
from skimage.io import imread


def get_data(path="./dog_dataset"):
    """
    :param path:
    :return: imgs - list of images
             nums - list of corresponding numerical breed-values
             dict{numerical breed: name of breed}
    """
    labels = os.listdir(path)
    labels.remove(".DS_Store")
    breeds = {}
    nums = []
    imgs = []
    for label in labels[:3]: # TODO: remove bounding
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

# imgs, nums, breeds = get_data()