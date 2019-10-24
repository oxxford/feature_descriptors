from scipy.ndimage.filters import convolve
import numpy as np


def gaussian_filter(sigma):
    size = 2 * np.ceil(3 * sigma) + 1
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) / (2 * np.pi * sigma ** 2)
    return g / g.sum()


def generate_octave(init_level, img_count_in_octave, sigma):
    octave = [init_level]

    k = 2 ** (1 / img_count_in_octave)
    kernel = gaussian_filter(k * sigma)

    for i in range(img_count_in_octave + 2):
        next_level = convolve(octave[-1], kernel)
        octave.append(next_level)

    return octave


def generate_gaussian_pyramid(im, num_octave, img_count_in_octave, sigma):
    pyr = []

    for _ in range(num_octave):
        octave = generate_octave(im, img_count_in_octave, sigma)
        pyr.append(octave)
        # s+3 images per octave -> third to last image as the base for the next octave
        # because it's the one with a blur of 2*sigma
        im = octave[-3][::2, ::2]

    return pyr


def generate_DoG_octave(gaussian_octave):
    octave = []

    for i in range(1, len(gaussian_octave)):
        octave.append(gaussian_octave[i] - gaussian_octave[i - 1])

    return np.concatenate([o[:, :, np.newaxis] for o in octave], axis=2)


def generate_DoG_pyramid(gaussian_pyramid):
    pyr = []

    for gaussian_octave in gaussian_pyramid:
        pyr.append(generate_DoG_octave(gaussian_octave))

    return pyr
