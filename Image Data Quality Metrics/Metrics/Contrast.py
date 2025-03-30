import cv2
import numpy as np


def michelson(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    I_max = np.max(gray)
    I_min = np.min(gray)
    return (I_max - I_min) / (I_max + I_min + 1e-8)


def rms(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.sqrt(np.mean(np.square(gray - np.mean(gray))))
