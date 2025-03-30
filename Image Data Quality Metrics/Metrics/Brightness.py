import cv2
import numpy as np


def mean(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def y_channel(image):
    yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    return np.mean(yuv[:, :, 0])


def hsv_value_mean(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return np.mean(hsv[:, :, 2])
