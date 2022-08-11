import cv2
import numpy as np
from ftcscore.util.lines import intersection, lines_to_distances


lsd = cv2.createLineSegmentDetector(scale=0.15)
# TODO: Distortion correction to make this work better


def detect_field(frame):
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
