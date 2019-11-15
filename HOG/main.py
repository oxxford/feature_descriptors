from skimage.io import imread
from HOG.hog_classifier import HOG
import json

hog = HOG()

# imgs, nums, breeds = get_data(path="../dog_dataset")
# hog.train(imgs, nums)
# hog.save('/Users/alsuvakhitova/Documents/Education/BS4/Fall/СV/Project/feature_descriptors/HOG/svm_data.dat')

with open("breeds.json") as f:
    breeds_json = f.read()
json_acceptable_string = breeds_json.replace("'", "\"")
breeds = json.loads(json_acceptable_string)
hog.load("/Users/alsuvakhitova/Documents/Education/BS4/Fall/СV/Project/feature_descriptors/HOG/svm_halfdata_train.dat")

img1 = imread("ts1.jpg")  # silky
img2 = imread("ts2.jpg")  # deerhound
img3 = imread("ts3.jpg")  # bay
img4 = imread("ts4.jpg")  # kerry blue

a = [img1, img2, img3, img4]

res = hog.predict([img1, img2, img3, img4])
breed_res = [breeds[str(int(n))] for n in res]

print(breed_res)