import cv2
import numpy as np


def rgb_average(image):
    b, g, r = cv2.split(image)
    return np.mean(b), np.mean(g), np.mean(r)