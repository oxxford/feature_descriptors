import pickle
import sys, os
from HOG.hog_classifier import HOG

def save_as_pickled_object(obj, filepath):
    """
    This is a defensive way to write pickle.write, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    bytes_out = pickle.dumps(obj)
    n_bytes = sys.getsizeof(bytes_out)
    with open(filepath, 'wb') as f_out:
        for idx in range(0, n_bytes, max_bytes):
            f_out.write(bytes_out[idx:idx+max_bytes])


def try_to_load_as_pickled_object_or_None(filepath):
    """
    This is a defensive way to write pickle.load, allowing for very large files on all platforms
    """
    max_bytes = 2**31 - 1
    try:
        input_size = os.path.getsize(filepath)
        bytes_in = bytearray(0)
        with open(filepath, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                bytes_in += f_in.read(max_bytes)
        obj = pickle.loads(bytes_in)
    except:
        return None
    return obj

# imgs, nums, breeds = get_data(path="../dog_dataset")
#
# c = list(zip(imgs, nums))
# random.shuffle(c)
# imgs, nums = zip(*c)
#
# half_len = int(0.5*len(nums))
# imgs_train, imgs_test, nums_train, nums_test = train_test_split(imgs[:half_len], nums[:half_len], test_size=0.2, random_state=42)
# print("# samples in train set: {}".format(len(imgs_train)))
#
# save_as_pickled_object(imgs_train, "imgs_train.pkl")
# save_as_pickled_object(imgs_test, "imgs_test.pkl")
# save_as_pickled_object(nums_train, "nums_train.pkl")
# save_as_pickled_object(nums_test, "nums_test.pkl")

hog = HOG()
hog.load("/Users/alsuvakhitova/Documents/Education/BS4/Fall/СV/Project/feature_descriptors/HOG/svm_halfdata_train.dat")
imgs_test = try_to_load_as_pickled_object_or_None("imgs_test.pkl")
nums_test = try_to_load_as_pickled_object_or_None("nums_test.pkl")

res = hog.predict(imgs_test)

accuracy=0
for i in range(len(res)):
  if res[i]==nums_test[i]:
    accuracy+=1
accuracy = accuracy/len(res)

print(0)

# with open('/Users/alsuvakhitova/Documents/Education/BS4/Fall/СV/Project/feature_descriptors/HOG/imgs_train.pkl', 'wb') as fp:
#     pickle.dump(imgs_train, fp)
# with open('/Users/alsuvakhitova/Documents/Education/BS4/Fall/СV/Project/feature_descriptors/HOG/imgs_test.pkl', 'wb') as fp:
#     pickle.dump(imgs_test, fp)
# with open('/Users/alsuvakhitova/Documents/Education/BS4/Fall/СV/Project/feature_descriptors/HOG/nums_train.pkl', 'wb') as fp:
#     pickle.dump(nums_train, fp)
# with open('/Users/alsuvakhitova/Documents/Education/BS4/Fall/СV/Project/feature_descriptors/HOG/nums_test.pkl', 'wb') as fp:
#     pickle.dump(nums_test, fp)