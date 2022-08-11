import cv2
import numpy as np
from ftcscore.util.lines import intersection, lines_to_distances


def detect_field(frame):
    mask = cv2.inRange(frame, (130,) * 3, (170,) * 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (100, 100))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = tuple(filter(lambda c: cv2.contourArea(c) > 1000, contours))

    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(mask_color, contours, -1, (0, 255, 0), thickness=3)
    m = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(m)
    rect = cv2.boundingRect(m)
    poly = cv2.approxPolyDP(m, 0.1 * cv2.arcLength(m, True), True)

    return mask_color, hull, rect, poly


lsd = cv2.createLineSegmentDetector(scale=0.15)
# TODO: Distortion correction to make this work better


def detect_field_edges(frame):
    def preprocess(inp):
        mask = cv2.GaussianBlur(inp, (5, 5), 1)

        mask = cv2.inRange(mask, (0,) * 3, (80,) * 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        dilate_hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 2))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, dilate_hor_kernel)

        dilate_vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 4))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, dilate_vert_kernel)

        return mask

    def detect_edges(inp):
        lines = lsd.detect(inp)[0]

        distances = lines_to_distances(lines)
        lines = lines[distances > 300]

        upper_line = min(lines, key=lambda l: l[0][1] + l[0][3])
        lower_line = max(lines, key=lambda l: l[0][1] + l[0][3])
        left_line = min(lines, key=lambda l: l[0][0] + l[0][2])
        right_line = max(lines, key=lambda l: l[0][0] + l[0][2])

        return np.array([upper_line, lower_line, left_line, right_line])

    def get_points(edges):
        upper, lower, left, right = edges
        ul = intersection(upper, left)
        ur = intersection(upper, right)
        ll = intersection(lower, left)
        lr = intersection(lower, right)

        return np.array([ul, ur, ll, lr])

    p = preprocess(frame)
    ls = detect_edges(p)
    pts = get_points(ls)

    mask_color = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)
    lsd.drawSegments(mask_color, ls)

    for point in pts:
        cv2.circle(mask_color, point, 5, (0, 255, 0), thickness=-1)

    return mask_color, pts


def detect_field_corners(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(frame, 4, 0.999, 200).astype('int32')

    for i in corners:
        x, y = i.ravel()
        cv2.circle(frame, (x, y), 10, (255, 255, 0), -1)

    return frame

# OpenCV contour page has lots of options
# Fit triangle to contour
# Simplify contour
# approx poly DP
# Canny edge detection
# Background detection might be usable
# Process video then try some stuff?
