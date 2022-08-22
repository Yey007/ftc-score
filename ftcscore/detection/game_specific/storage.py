import cv2
import numpy as np
from ftcscore.processing.normalize import normalize_lcn
from ftcscore.util.contours import contour_x, contour_y, contour_aspect_ratio


def detect_alliance_hub(frame, lowerb, upperb):
    norm = normalize_lcn(frame)
    mask = cv2.inRange(norm, lowerb, upperb)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = filter(lambda c: cv2.contourArea(c) > 1000, contours)
    contours = filter(lambda c: 1.4 > contour_aspect_ratio(c) > 0.6, contours)
    top_contour = min(contours, key=contour_y)
    top_x = contour_x(top_contour)

    mask = cv2.inRange(frame, (0, 0, 0), (40, 40, 40))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = filter(lambda c: top_x - 20 <= contour_x(c) <= top_x + 20, contours)
    bottom_contour = max(contours, key=cv2.contourArea)

    both = np.concatenate(np.array([top_contour, bottom_contour], dtype='object'))

    rect = cv2.boundingRect(both)
    return rect


def detect_alliance_hub_blue(frame):
    return detect_alliance_hub(frame, (40, 0, 0), (130, 40, 40))


def detect_alliance_hub_red(frame):
    return detect_alliance_hub(frame, (0, 0, 40), (20, 20, 110))


def detect_shared_hub(overhead):
    grey = cv2.cvtColor(overhead, cv2.COLOR_BGR2GRAY)
    grey = cv2.GaussianBlur(grey, (11, 11), 1)

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
    circles = list(circles)

    if len(circles) == 0:
        return None

    x, y, r = np.uint16(max(circles, key=lambda c: c[2]))
    rect = x - r, y - r, 2 * r, 2 * r

    return rect
