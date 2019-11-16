from commons.custom_pickle_loader import try_to_load_as_pickled_object_or_None
import numpy as np
from hog_classifier import hog_desc
import pickle

"""
Here we extract and describe features from images using HOG algorithm
The feature vectors are saved for later training
"""

imgs_train = try_to_load_as_pickled_object_or_None("../imgs_train.pkl")
imgs_test = try_to_load_as_pickled_object_or_None("../imgs_test.pkl")

# extracting train images features
features_train = []
for img in imgs_train:
  features_train.append(np.float32(hog_desc(img)))
features_train = np.array(features_train)

# extracting test images features
features_test = []
for img in imgs_test:
  features_test.append(np.float32(hog_desc(img)))
features_test = np.array(features_test)


# saving everything
with open('hog_features_train.pkl', 'wb') as f:
  pickle.dump(features_train, f)

with open('hog_features_test.pkl', 'wb') as f:
  pickle.dump(features_test, f)

