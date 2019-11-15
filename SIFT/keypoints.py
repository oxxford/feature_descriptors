"""
Second step of SIFT. Finding and localizing the keypoints.
"""

import numpy as np
import numpy.linalg as LA


def get_candidate_keypoints(octave, w=16):
    # w is side length for the patches used when creating local descriptors
    candidates = []

    octave[:, :, 0] = 0
    octave[:, :, -1] = 0

    # have to start at w//2 so that when getting the local w x w descriptor, we don't fall off
    for i in range(w // 2 + 1, octave.shape[0] - w // 2 - 1):
        for j in range(w // 2 + 1, octave.shape[1] - w // 2 - 1):
            for k in range(1, octave.shape[2] - 1):
                patch = octave[i - 1:i + 2, j - 1:j + 2, k - 1:k + 2]
                if np.argmax(patch) == 13 or np.argmin(patch) == 13:
                    candidates.append([i, j, k])

    return candidates


def localize_keypoint(octave, x, y, octave_img_count):
    # computing first derivatives with Jacobian
    dx = (octave[y, x + 1, octave_img_count] - octave[y, x - 1, octave_img_count]) / 2.
    dy = (octave[y + 1, x, octave_img_count] - octave[y - 1, x, octave_img_count]) / 2.
    ds = (octave[y, x, octave_img_count + 1] - octave[y, x, octave_img_count - 1]) / 2.

    # computing second derivatives with Hessian
    dxx = octave[y, x + 1, octave_img_count] - 2 * octave[y, x, octave_img_count] + octave[y, x - 1, octave_img_count]
    dxy = ((octave[y + 1, x + 1, octave_img_count] - octave[y + 1, x - 1, octave_img_count]) - (octave[y - 1, x + 1, octave_img_count] - octave[y - 1, x - 1, octave_img_count])) / 4.
    dxs = ((octave[y, x + 1, octave_img_count + 1] - octave[y, x - 1, octave_img_count + 1]) - (octave[y, x + 1, octave_img_count - 1] - octave[y, x - 1, octave_img_count - 1])) / 4.
    dyy = octave[y + 1, x, octave_img_count] - 2 * octave[y, x, octave_img_count] + octave[y - 1, x, octave_img_count]
    dys = ((octave[y + 1, x, octave_img_count + 1] - octave[y - 1, x, octave_img_count + 1]) - (octave[y + 1, x, octave_img_count - 1] - octave[y - 1, x, octave_img_count - 1])) / 4.
    dss = octave[y, x, octave_img_count + 1] - 2 * octave[y, x, octave_img_count] + octave[y, x, octave_img_count - 1]

    jacobian = np.array([dx, dy, ds])
    harris_detector = np.array([
        [dxx, dxy, dxs],
        [dxy, dyy, dys],
        [dxs, dys, dss]])

    offset = -LA.inv(harris_detector).dot(jacobian)
    return offset, jacobian, harris_detector[:2, :2], x, y, octave_img_count


def find_keypoints_for_DoG_octave(octave, threshold_R, threshold_contrast, w):
    candidates = get_candidate_keypoints(octave, w)

    keypoints = []

    for i, candidate in enumerate(candidates):
        y, x, octave_img_count = candidate[0], candidate[1], candidate[2]
        offset, jacobian, harris_detector, x, y, octave_img_count = localize_keypoint(octave, x, y, octave_img_count)

        contrast = octave[y, x, octave_img_count] + .5 * jacobian.dot(offset)
        if abs(contrast) < threshold_contrast: continue

        w, v = LA.eig(harris_detector)
        r = w[1] / w[0]
        R = (r + 1) ** 2 / r
        if R > threshold_R: continue

        cand_keyopints = np.array([x, y, octave_img_count]) + offset
        # throw out boundary points
        if cand_keyopints[1] >= octave.shape[0] or cand_keyopints[0] >= octave.shape[1]: continue

        keypoints.append(cand_keyopints)

    return np.array(keypoints)


def get_keypoints(DoG_pyramid, threshold_R, threshold_contrast, w):
    kps = []

    for D in DoG_pyramid:
        kps.append(find_keypoints_for_DoG_octave(D, threshold_R, threshold_contrast, w))

    return kps
