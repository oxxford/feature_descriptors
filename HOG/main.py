from skimage.io import imread, imshow
from skimage.transform import resize
from skimage import exposure
import matplotlib.pyplot as plt
import numpy as np
from HOG.hog_classifier import HOG
from dataset_parser import get_data
import cv2

imgs, nums, breeds = get_data(path="../dog_dataset")
hog = HOG()
hog.train(imgs, nums)

img1 = imread("ts1.jpg")  # silky
img2 = imread("ts2.jpg")  # deerhound
img3 = imread("ts3.jpg")  # bay

res = hog.predict([img1, img2, img3])
breed_res = [breeds[n] for n in res]

print(breed_res)