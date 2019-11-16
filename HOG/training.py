from commons.custom_pickle_loader import try_to_load_as_pickled_object_or_None
from commons.svm import SVM
import numpy as np

features_train = try_to_load_as_pickled_object_or_None("features_train.pkl")
features_test = try_to_load_as_pickled_object_or_None("features_test.pkl")
nums_train = try_to_load_as_pickled_object_or_None("../main_data/nums_train.pkl")
nums_test = try_to_load_as_pickled_object_or_None("../main_data/nums_test.pkl")

def train_svm(svm, descriptors, labels, path_to_save='svm_halfdata_train.dat'):
    train_data = np.matrix(descriptors, dtype=np.float32)
    labels = np.array([labels]).transpose()
    svm.train(train_data, labels)
    svm.save(path_to_save)
    return svm

def predict_svm(svm, test_descriptors):
    predict_data = np.matrix(test_descriptors, dtype=np.float32)
    return svm.predict(predict_data)

svm = SVM()
train_svm(svm, features_train, nums_train)

loaded_svm = SVM()
loaded_svm.load('svm_halfdata_train.dat')
res = svm.predict(features_test)

accuracy=0
for i in range(len(res)):
  if int(res[i])==nums_test[i]:
    accuracy+=1
accuracy = accuracy/len(res)

print("Accuracy: {0:.2f}%".format(accuracy*100))
