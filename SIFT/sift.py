"""
Combining all steps together
"""

from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve

import SIFT.DoG_pyramid as pyramid
from SIFT.keypoints import get_keypoints
from SIFT.orientation import assign_orientation
from SIFT.descriptors import get_local_descriptors


class SIFT(object):
    def __init__(self, im, s=3, num_octave=4, s0=1.3, sigma=1.6, r_th=10, t_c=0.03, w=16):
        self.im = convolve(rgb2gray(im), pyramid.gaussian_filter(s0))
        self.s = s
        self.sigma = sigma
        self.num_octave = num_octave
        self.t_c = t_c
        self.R_th = (r_th+1)**2 / r_th
        self.w = w

    def get_features(self):
        gaussian_pyr = pyramid.generate_gaussian_pyramid(self.im, self.num_octave, self.s, self.sigma)
        DoG_pyr = pyramid.generate_DoG_pyramid(gaussian_pyr)
        kp_pyr = get_keypoints(DoG_pyr, self.R_th, self.t_c, self.w)
        feats = []

        for i, DoG_octave in enumerate(DoG_pyr):
            kp_pyr[i] = assign_orientation(kp_pyr[i], DoG_octave)
            feats.append(get_local_descriptors(kp_pyr[i], DoG_octave))

        self.kp_pyr = kp_pyr
        self.feats = feats

        return feats
