import cv2
import numpy as np

from ftcscore.processing.normalize import normalize_lcn, normalize_comprehensive, normalize_standard
from ftcscore.util.contours import contour_y, contour_x, contour_aspect_ratio

open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (16, 16))

tracker_blue = None
tracker_red = None
tracker_shared = None


# TODO: change hardcoded pixel thresholds to percentage of image or smth


def detect_alliance_hub_blue(frame):
    global tracker_blue

    # https://pyimagesearch.com/2018/07/30/opencv-object-tracking/
    if tracker_blue is not None:
        (success, box) = tracker_blue.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
            return

    norm = normalize_lcn(frame)
    mask = cv2.inRange(norm, (40, 0, 0), (130, 40, 40))
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
    tracker_blue = cv2.legacy.TrackerKCF_create()
    tracker_blue.init(frame, rect)


def detect_alliance_hub_red(frame):
    global tracker_red

    # https://pyimagesearch.com/2018/07/30/opencv-object-tracking/
    if tracker_red is not None:
        (success, box) = tracker_red.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
            return

    norm = normalize_lcn(frame)
    mask = cv2.inRange(norm, (0, 0, 40), (20, 20, 110))
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
    tracker_red = cv2.legacy.TrackerKCF_create()
    tracker_red.init(frame, rect)


def detect_shared_hub(frame):
    global tracker_shared

    # https://pyimagesearch.com/2018/07/30/opencv-object-tracking/
    if tracker_shared is not None:
        (success, box) = tracker_shared.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)
            return

    mask = cv2.inRange(frame, (0,) * 3, (80,) * 3)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = filter(lambda c: frame.shape[1] // 2 - 40 < contour_x(c) < frame.shape[1] // 2 + 40, contours)
    contours = filter(lambda c: contour_y(c) > frame.shape[0] // 2, contours)
    bottom_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(bottom_contour)
    rect = (x, y, w, h - 25)
    tracker_shared = cv2.legacy.TrackerKCF_create()
    tracker_shared.init(frame, rect)
