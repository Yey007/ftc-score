import cv2
from ftcscore.processing.normalize import normalize_lcn

open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))


def contour_center(c):
    moments = cv2.moments(c)
    if moments['m00'] != 0:
        return moments['m10'] / moments['m00']
    else:
        return float('inf')


def detect_alliance_hub_blue(frame):
    norm = normalize_lcn(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (110, 115, 60), (125, 255, 200))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)

    contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # Sort by area desc, center asc
    contours = sorted(contours, key=contour_center)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[:2]

    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(mask_color, contours, -1, color=(0, 255, 0))

    for contour in contours:
        ellipse = cv2.fitEllipse(contour)
        cv2.ellipse(frame, ellipse, (0, 255, 0), 2)

    cv2.imshow('norm', norm)
    cv2.imshow('mask', mask_color)
