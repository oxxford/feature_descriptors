import cv2
import numpy as np

class StatModel(object):
    '''parent class - starting point to add abstraction'''
    def load(self, fn):
        self.model.load(fn)
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    '''wrapper for OpenCV SimpleVectorMachine algorithm'''
    def __init__(self, kernel = "linear"):
        self.model = cv2.ml.SVM_create()
        self.kernel = kernel

    def train(self, samples, responses):
        """
        Training SVM
        :param samples: np matrix of np arrays, type - float32
        :param responses: np array of classes
        :return:
        """
        #setting algorithm parameters
        self.model.setType(cv2.ml.SVM_C_SVC)
        if self.kernel == "linear":
            self.model.setKernel(cv2.ml.SVM_LINEAR)
        elif self.kernel == "chi":
            self.model.setKernel(cv2.ml.SVM_CHI2)
        elif self.kernel == "inter":
            self.model.setKernel(cv2.ml.SVM_INTER)
        elif self.kernel == "rbf":
            self.model.setKernel(cv2.ml.SVM_RBF)
        elif self.kernel == "sigmoid":
            self.model.setKernel(cv2.ml.SVM_SIGMOID)
        else:
            print("Error: Invalid kernel name")

        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        """
        This function is used to predict class of sample using trained SVM
        :param samples: nparray of feature vectors (np arrays, float32 type)
        :return: list of predicted classes
        """
        res = []
        for sample in samples:
            pr = self.model.predict(np.array([sample]))
            pr = pr[1][0][0]
            res.append(pr)
        return res

    def load(self, path):
        """
        Loading of pre-trained model
        :param path: path to model
        :return:
        """
        self.model = cv2.ml.SVM_load(path)

    def save(self, path):
        """
        Saving trined model
        :param path: path to save the model
        :return:
        """
        self.model.save(path)