import numpy as np
from scipy.spatial import distance


def is_keypoint(image, row, col, threshold):
    """
    Checks if a pixel at (row, col) is a potential keypoint
    :param image: original image
    :param row: row position of the pixel
    :param col: column position of the pixel
    :param threshold: threshold to compare the difference with
    :return: boolean value if a pixel is a keypoint
    """
    intensity = int(image[row][col])

    intensity1 = int(image[row - 3][col])
    intensity9 = int(image[row + 3][col])
    intensity5 = int(image[row][col + 3])
    intensity13 = int(image[row][col - 3])

    count = 0
    if np.abs(intensity1 - intensity) > threshold:
        count += 1
    if np.abs(intensity9 - intensity) > threshold:
        count += 1
    if np.abs(intensity5 - intensity) > threshold:
        count += 1
    if np.abs(intensity13 - intensity) > threshold:
        count += 1

    return count >= 3


def calculate_score(image, point):
    """
    Calculates score of a point in an image
    :param image:
    :param point:
    :return: integer value with the score
    """
    col, row = point

    intensity = int(image[row][col])
    intensity1 = int(image[row + 3][col])
    intensity3 = int(image[row + 2][col + 2])
    intensity5 = int(image[row][col + 3])
    intensity7 = int(image[row - 2][col + 2])
    intensity9 = int(image[row - 3][col])
    intensity11 = int(image[row + 2][col - 2])
    intensity13 = int(image[row][col - 3])
    intensity15 = int(image[row - 2][col - 2])

    score = np.abs(intensity - intensity1) + np.abs(intensity - intensity3) + \
            np.abs(intensity - intensity5) + np.abs(intensity - intensity7) + \
            np.abs(intensity - intensity9) + np.abs(intensity - intensity11) + \
            np.abs(intensity - intensity13) + np.abs(intensity - intensity15)

    return score


def suppress(image, keypoints):
    """
    Performs non-maxima supression
    :param image: original image
    :param keypoints: list of potential keypoints
    """
    i = 1
    while i < len(keypoints):
        curr = keypoints[i]
        prev = keypoints[i - 1]

        if distance.euclidean(curr[0] - prev[0], curr[1] - prev[1]) <= 4:
            curr_score = calculate_score(image, curr)
            prev_score = calculate_score(image, prev)

            if curr_score > prev_score:
                del (keypoints[i - 1])
            else:
                del (keypoints[i])
        else:
            i += 1
            continue


def detect(image, threshold=50):
    """
    Detects keypoint in the image using FAST
    :param image:
    :param threshold: value compare with
    :return: list of keypoints
    """
    keypoints = []
    rows, cols = image.shape

    for row in range(3, rows - 3):
        for col in range(3, cols - 3):
            if is_keypoint(image, row, col, threshold):
                keypoints.append((col, row))

    suppress(image, keypoints)
    return keypoints


def get_brief_description(imgray, features, patch_size=48, size=128):
    """
    Generates description of keypoints with BRIEF descriptor
    :param imgray: original image
    :param features: features to describe
    :param patch_size: size of the patch to draw points from
    :param size: size of the descriptor
    :return: list of descriptions for features
    """
    result = []

    for feature in features:
        if feature[0] < patch_size / 2 or feature[0] > imgray.shape[0] - patch_size / 2 or \
                feature[1] < patch_size / 2 or feature[1] > imgray.shape[1] - patch_size / 2:
            continue

        descriptor = []

        for i in range(size):
            first_x = feature[0] + np.random.randint(- patch_size // 2, patch_size // 2)
            first_y = feature[1] + np.random.randint(- patch_size // 2, patch_size // 2)

            second_x = feature[0] + np.random.randint(- patch_size // 2, patch_size // 2)
            second_y = feature[1] + np.random.randint(- patch_size // 2, patch_size // 2)

            if imgray[first_x][first_y] < imgray[second_x][second_y]:
                descriptor.append(0)
            else:
                descriptor.append(1)

        result.append(descriptor)

    return result
