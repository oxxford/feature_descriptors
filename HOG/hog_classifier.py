from skimage.transform import resize
from numpy.linalg import norm
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
    def __init__(self):
        self.model = cv2.ml.SVM_create()

    def train(self, samples, responses):
        #setting algorithm parameters
        self.model.setType(cv2.ml.SVM_C_SVC)
        self.model.setKernel(cv2.ml.SVM_LINEAR)
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        res = []
        for sample in samples:
            pr = self.model.predict(sample)
            pr = pr[1][0][0]
            res.append(pr)
        return res

class HOG(object):

    def __init__(self):
        self.clf = SVM()

    def __hog_desc(self, img, window_size = 8, stride = 4, bin_n = 9):
        proc_img = np.float32(resize(img, (128, 64)))
        proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)

        height = len(proc_img)
        width = len(proc_img[0])
        features = []
        for y in range(0, height - window_size, stride):
            for x in range(0, width - window_size, stride):
                wnd = proc_img[y:y + window_size, x:x + window_size]

                # Do edge detection.
                gx = cv2.Sobel(wnd, cv2.CV_32F, 1, 0)
                gy = cv2.Sobel(wnd, cv2.CV_32F, 0, 1)
                mag, ang = cv2.cartToPolar(gx, gy)

                # Bin the angles.
                bin = np.int32(bin_n * ang / (2 * np.pi))
                # the magnitudes are used as weights for the gradient values.
                hist = np.bincount(bin.ravel(), mag.ravel(), bin_n)

                # transform to Hellinger kernel
                eps = 1e-7
                hist /= hist.sum() + eps
                hist = np.sqrt(hist)
                hist /= norm(hist) + eps
                features.extend(hist)

        return np.float32(features)

    def train(self, images, labels):
        descriptors = []
        count = 0
        for img in images:
            ft = np.float32(self.__hog_desc(img))
            descriptors.append(ft)
            count += 1
            if count%50 == 0:
                print("Size of vector %d" % (len(ft)))
                print("Completed: %d" % (count))
        labels = np.array(labels)
        print(np.array(descriptors).shape)
        self.clf.train(np.array(descriptors), np.array([labels]).transpose())

    def predict(self, samples):
        features = []
        for sample in samples:
            ft = self.__hog_desc(sample)
            ft = np.transpose(np.array([ft]).transpose(), (1, 0))
            features.append(ft)
        return self.clf.predict(features)