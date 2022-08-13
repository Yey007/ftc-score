import cv2
import numpy as np

from ftcscore.processing.normalize import normalize_lcn

open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))


def contour_y(c):
    moments = cv2.moments(c)
    if moments['m00'] != 0:
        return moments['m01'] / moments['m00']
    else:
        return float('inf')


def contour_x(c):
    moments = cv2.moments(c)
    if moments['m00'] != 0:
        return moments['m10'] / moments['m00']
    else:
        return float('inf')


tracker = None


def detect_alliance_hub_blue(frame):
    global tracker

    # https://pyimagesearch.com/2018/07/30/opencv-object-tracking/
    if tracker is not None:
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
            return

    norm = normalize_lcn(frame)
    mask = cv2.inRange(norm, (40, 0, 0), (130, 40, 40))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = filter(lambda c: cv2.contourArea(c) > 100, contours)
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
    tracker = cv2.legacy.TrackerKCF_create()
    tracker.init(frame, rect)
