from skimage.io import imread
from SIFT.sift import SIFT
import matplotlib.pyplot as plt


image = imread('dog.jpeg')

sift_detector = SIFT(image)
sift_detector.get_features()
pyramid = sift_detector.kp_pyr
print(f'Number of features found: {sift_detector.feats[0].shape[0]}')
print(f'Array of features: {sift_detector.feats[0]}')

plt.imshow(image)
kps = pyramid[0]
plt.scatter(kps[:,0], kps[:,1], c='b', s=2.5)
plt.show()
