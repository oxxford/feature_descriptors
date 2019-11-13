"""
The last step of SIFT. Creating descriptors.
"""

import numpy as np
import numpy.linalg as LA


def get_histogram_for_subregion(m, theta, num_bin, reference_angle, bin_width, subregion_w):
    hist = np.zeros(num_bin, dtype=np.float32)
    c = subregion_w / 2 - .5

    for i, (magnitude, angle) in enumerate(zip(m, theta)):
        angle = (angle - reference_angle) % 360
        binno = int(np.floor(angle)//bin_width)
        vote = magnitude

        # binno*bin_width is the start angle of the histogram bin
        # binno*bin_width+bin_width/2 is the center of the histogram bin
        # angle - " is the distance from the angle to the center of the bin 
        hist_interp_weight = 1 - abs(angle - (binno * bin_width + bin_width / 2)) / (bin_width / 2)
        vote *= max(hist_interp_weight, 1e-6)

        gy, gx = np.unravel_index(i, (subregion_w, subregion_w))
        x_interp_weight = max(1 - abs(gx - c) / c, 1e-6)
        y_interp_weight = max(1 - abs(gy - c) / c, 1e-6)
        vote *= x_interp_weight * y_interp_weight

        hist[binno] += vote

    return hist


def get_local_descriptors(keypoints, octave, w=16, num_subregion=4, num_bin=8):
    descriptors = []
    bin_width = 360 // num_bin

    for kp in keypoints:
        cx, cy, s = int(kp[0]), int(kp[1]), int(kp[2])
        s = np.clip(s, 0, octave.shape[2] - 1)
        L = octave[..., s]

        t, l = max(0, cy - w // 2), max(0, cx - w // 2)
        b, r = min(L.shape[0], cy + w // 2 + 1), min(L.shape[1], cx + w // 2 + 1)
        patch = L[t:b, l:r]

        dx, dy = get_patch_gradients(patch)
        m = np.sqrt(dx ** 2 + dy ** 2)
        theta = (np.arctan2(dy, dx) + np.pi) * 180 / np.pi

        subregion_w = w // num_subregion
        featvec = np.zeros(num_bin * num_subregion ** 2, dtype=np.float32)

        for i in range(0, subregion_w):
            for j in range(0, subregion_w):
                t, l = i * subregion_w, j * subregion_w
                b, r = min(L.shape[0], (i + 1) * subregion_w), min(L.shape[1], (j + 1) * subregion_w)

                hist = get_histogram_for_subregion(m[t:b, l:r].ravel(),
                                                   theta[t:b, l:r].ravel(),
                                                   num_bin,
                                                   kp[3],
                                                   bin_width,
                                                   subregion_w)
                featvec[i * subregion_w * num_bin + j * num_bin:i * subregion_w * num_bin + (j + 1) * num_bin] = hist.flatten()

        featvec /= max(1e-6, LA.norm(featvec))
        featvec[featvec > 0.2] = 0.2
        featvec /= max(1e-6, LA.norm(featvec))
        descriptors.append(featvec)

    return np.array(descriptors)


def get_patch_gradients(patch):
    temp = np.zeros_like(patch)
    temp[-1] = patch[-1]
    temp[:-1] = patch[1:]
    temp2 = np.zeros_like(patch)
    temp2[0] = patch[0]
    temp2[1:] = patch[:-1]

    dy = temp - temp2

    temp[:, -1] = patch[:, -1]
    temp[:, :-1] = patch[:, 1:]
    temp2[:, 0] = patch[:, 0]
    temp2[:, 1:] = patch[:, :-1]

    dx = temp - temp2

    return dx, dy
