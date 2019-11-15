from skimage.transform import resize
from numpy.linalg import norm
import cv2
import numpy as np
from commons.svm import SVM

class HOG(object):

    def __init__(self):
        self.clf = SVM()

    def __hog_desc(self, img, window_size = 8, stride = 4, bin_n = 9):

        height = 128
        width = 64

        proc_img = np.float32(resize(img, (height, width)))
        proc_img = cv2.cvtColor(proc_img, cv2.COLOR_BGR2GRAY)

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

        if len(features) != 3780:
            if len(features) > 3780:
                features = features[:3780]
            if len(features) < 3780:
                features += [0] * (3780 - len(features))

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
        train_data = np.matrix(descriptors, dtype=np.float32)
        labels = np.array([labels]).transpose()
        self.clf.train(train_data, labels)

    def predict(self, samples):
        features = []
        for sample in samples:
            ft = np.float32(self.__hog_desc(sample))
            features.append(ft)
        predict_data = np.matrix(features, dtype=np.float32)
        return self.clf.predict(predict_data)


    def save(self, path):
        self.clf.save(path)

    def load(self, path):
        self.clf.load(path)