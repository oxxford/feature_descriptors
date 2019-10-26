from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
from HOG.hog_descr import hog_desc, train_hog
from dataset_parser import get_data
import cv2

imgs, nums, breeds = get_data(path="../dog_dataset")
clf = train_hog(imgs, nums)

img = imread("ts1.jpg")  # silky
ft = np.float32(hog_desc(img))
a = np.transpose(np.array([ft]).transpose(), (1,0))

img = imread("ts2.jpg")  # deerhound
ft = np.float32(hog_desc(img))
b = np.transpose(np.array([ft]).transpose(), (1,0))

img = imread("ts3.jpg")  # bay
ft = np.float32(hog_desc(img))
c = np.transpose(np.array([ft]).transpose(), (1,0))

res = clf.predict([a,b,c])
breed_res = [breeds[n[1]] for n in res]

print(9)