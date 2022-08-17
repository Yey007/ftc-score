import cv2
import numpy as np


def detect_shared_hub(overhead):
    grey = cv2.cvtColor(overhead, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (5, 5), 1)

    circles = cv2.HoughCircles(grey, cv2.HOUGH_GRADIENT, 1, 20,
                               param1=100, param2=50, minRadius=10, maxRadius=50)

    if circles is None:
        return None

    circles = circles[0, :]

    # Bottom half and middle third
    half_height = overhead.shape[1] // 2
    third_width = overhead.shape[0] // 3
    circles = filter(lambda c: c[1] > half_height, circles)
    circles = filter(lambda c: third_width < c[0] < 2 * third_width, circles)

    return np.uint0(max(circles, key=lambda c: c[2]))
