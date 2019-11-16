import random
from sklearn.model_selection import train_test_split
from commons.custom_pickle_loader import save_as_pickled_object
from commons.dataset_parser import get_data
import json

"""
Here we split half of original dataset into train and test sets
(we use half because of computation limitations)
"""

imgs, nums, breeds = get_data(path="../dog_dataset")

# shuffling original data
c = list(zip(imgs, nums))
random.shuffle(c)
imgs, nums = zip(*c)

# splitting
half_len = int(0.5*len(nums))
imgs_train, imgs_test, nums_train, nums_test = train_test_split(imgs[:half_len], nums[:half_len], test_size=0.2, random_state=42)
print("# samples in train set: {}".format(len(imgs_train)))

# saving
save_as_pickled_object(imgs_train, "../main_data/imgs_train.pkl")
save_as_pickled_object(imgs_test, "../main_data/imgs_test.pkl")
save_as_pickled_object(nums_train, "../main_data/nums_train.pkl")
save_as_pickled_object(nums_test, "../main_data/nums_test.pkl")

with open("breeds.json") as f:
    json.dumps(breeds)