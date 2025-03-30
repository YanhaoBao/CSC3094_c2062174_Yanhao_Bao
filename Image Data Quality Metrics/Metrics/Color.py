import cv2
import numpy as np


def rgb_average(image):
    b, g, r = cv2.split(image)
    return np.mean(b), np.mean(g), np.mean(r)


def histogram(image, bins=256):
    chans = cv2.split(image)
    return [cv2.calcHist([ch], [0], None, [bins], [0, 256]) for ch in chans]
