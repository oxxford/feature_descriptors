from commons.custom_pickle_loader import try_to_load_as_pickled_object_or_None
from commons.svm import SVM

"""
Here we train five SVMs (different kernels)

For CatBoost testing see: https://colab.research.google.com/drive/1eFhINcjZka73pFHMD1u5c_oeH7nN3Otk
"""

def test_svm():
  features_test = try_to_load_as_pickled_object_or_None("hog_features_test.pkl")
  nums_test = try_to_load_as_pickled_object_or_None("../main_data/nums_test.pkl")

  kernels = ["linear", "chi", "inter", "rbf", "sigmoid"]

  for kr in kernels:
    path = './models/svm_halfdata_train_' + kr + '.dat'

    loaded_svm = SVM()
    loaded_svm.load(path)
    res = loaded_svm.predict(features_test)

    accuracy=0
    for i in range(len(res)):
      if int(res[i])==nums_test[i]:
        accuracy+=1
    accuracy = accuracy/len(res)

    print("Using SVM with {} kernel".format(kr))
    print("Accuracy: {0:.2f}%".format(accuracy*100))

if __name__ == '__main__':
    test_svm()