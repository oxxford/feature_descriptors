import numpy as np
from scipy.spatial import distance


def is_corner(image, row, col, threshold):
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


def suppress(image, corners):
    i = 1
    while i < len(corners):
        curr = corners[i]
        prev = corners[i - 1]

        if distance.euclidean(curr[0] - prev[0], curr[1] - prev[1]) <= 4:
            curr_score = calculate_score(image, curr)
            prev_score = calculate_score(image, prev)

            if curr_score > prev_score:
                del (corners[i - 1])
            else:
                del (corners[i])
        else:
            i += 1
            continue
    return


def detect(image, threshold=50):
    corners = []
    rows, cols = image.shape

    for row in range(3, rows - 3):
        for col in range(3, cols - 3):
            if is_corner(image, row, col, threshold):
                corners.append((col, row))

    suppress(image, corners)
    return corners
