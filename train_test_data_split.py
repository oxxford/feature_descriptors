import random
from sklearn.model_selection import train_test_split
from commons.custom_pickle_loader import save_as_pickled_object
from commons.dataset_parser import get_data
import json

"""
Here we split half of original dataset into train and test sets
(we use half because of computation limitations)
"""

def main():
    imgs, nums, breeds = get_data(path="./dog_dataset")

    # here we count how many images are in first 60 classes
    # sa that later we can slice arrays with that number
    new=-1
    unique=0
    count=0
    for i in range(len(nums)):
        if nums[i]!=new:  # encountered new class
            if unique == 60:  # already have 60 unique classes
                break
            else:
                new = nums[i]
                unique += 1
        count+=1

    print("{} samples will be used".format(count))

    # shuffling original data
    c = list(zip(imgs[:count], nums[:count]))
    random.shuffle(c)
    imgs, nums = zip(*c)

    # splitting
    imgs_train, imgs_test, nums_train, nums_test = train_test_split(imgs, nums, test_size=0.2, random_state=42)
    print("# samples in train set: {}".format(len(imgs_train)))

    # saving
    save_as_pickled_object(imgs_train, "./main_data/imgs_train.pkl")
    save_as_pickled_object(imgs_test, "./main_data/imgs_test.pkl")
    save_as_pickled_object(nums_train, "./main_data/nums_train.pkl")
    save_as_pickled_object(nums_test, "./main_data/nums_test.pkl")

    with open("./main_data/breeds.json", 'w') as f:
        json.dump(breeds, f)

if __name__ == '__main__':
    main()