"""
Third step of SIFT. Assigning orientation to keypoints.
"""

import numpy as np
from numpy import linalg as LA

from SIFT.DoG_pyramid import gaussian_filter


def get_gradient(L, x, y):
    dy = L[min(L.shape[0] - 1, y + 1), x] - L[max(0, y - 1), x]
    dx = L[y, min(L.shape[1] - 1, x + 1)] - L[y, max(0, x - 1)]

    r = np.sqrt(dx ** 2 + dy ** 2)
    theta = (np.arctan2(dy, dx) + np.pi) * 180 / np.pi
    return r, theta


def fit_parabola(hist, bin_number, bin_width):
    centerval = bin_number * bin_width + bin_width / 2.

    if bin_number == len(hist) - 1:
        rightval = 360 + bin_width / 2.
    else:
        rightval = (bin_number + 1) * bin_width + bin_width / 2.

    if bin_number == 0:
        leftval = -bin_width / 2.
    else:
        leftval = (bin_number - 1) * bin_width + bin_width / 2.

    A = np.array([
        [centerval ** 2, centerval, 1],
        [rightval ** 2, rightval, 1],
        [leftval ** 2, leftval, 1]])
    b = np.array([
        hist[bin_number],
        hist[(bin_number + 1) % len(hist)],
        hist[(bin_number - 1) % len(hist)]])

    x = LA.lstsq(A, b, rcond=None)[0]
    if x[0] == 0: x[0] = 1e-6
    return -x[1] / (2 * x[0])


def assign_orientation(keypoints, octave, num_bins=36):
    new_keypoints = []
    bin_width = 360 // num_bins

    for keypoint in keypoints:
        cx, cy, s = int(keypoint[0]), int(keypoint[1]), int(keypoint[2])
        s = np.clip(s, 0, octave.shape[2] - 1)

        sigma = keypoint[2] * 1.5
        w = int(2 * np.ceil(sigma) + 1)
        kernel = gaussian_filter(sigma)

        L = octave[..., s]
        hist = np.zeros(num_bins, dtype=np.float32)

        for oy in range(-w, w + 1):
            for ox in range(-w, w + 1):
                x, y = cx + ox, cy + oy

                if x < 0 or x > octave.shape[1] - 1:
                    continue
                elif y < 0 or y > octave.shape[0] - 1:
                    continue

                m, theta = get_gradient(L, x, y)
                weight = kernel[oy + w, ox + w] * m

                bin = int(np.floor(theta) // (360 // num_bins))
                hist[bin] += weight

        max_bin = np.argmax(hist)
        new_keypoints.append([keypoint[0], keypoint[1], keypoint[2], fit_parabola(hist, max_bin, bin_width)])

        max_val = np.max(hist)
        for binno, val in enumerate(hist):
            if binno == max_bin: continue

            if .8 * max_val <= val:
                new_keypoints.append([keypoint[0], keypoint[1], keypoint[2], fit_parabola(hist, binno, bin_width)])

    return np.array(new_keypoints)
