import cv2
import numpy as np

from ftcscore.processing.normalize import normalize_comprehensive
from ftcscore.util.lines import intersection, lines_to_distances

lsd = cv2.createLineSegmentDetector(scale=0.1)


# TODO: Distortion correction to make this work better


def detect_field_from_edges(frame):
    def preprocess(inp):
        mask = cv2.GaussianBlur(inp, (5, 5), 1)

        mask = cv2.inRange(mask, (0,) * 3, (80,) * 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
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

        important_lines = np.array([upper_line, lower_line, left_line, right_line])

        return important_lines, lines

    def get_points(edges):
        upper, lower, left, right = edges
        ul = intersection(upper, left)
        ur = intersection(upper, right)
        ll = intersection(lower, left)
        lr = intersection(lower, right)

        return np.array([ul, ur, ll, lr])

    p = preprocess(frame)
    ls, all_ls = detect_edges(p)

    mask_color = cv2.cvtColor(p, cv2.COLOR_GRAY2BGR)

    for line in all_ls:
        line = line[0].astype('int32')
        cv2.line(mask_color, line[:2], line[2:], color=(255, 0, 0), thickness=3)

    for line in ls:
        line = line[0].astype('int32')
        cv2.line(mask_color, line[:2], line[2:], color=(0, 255, 0), thickness=3)

    u = np.unique(ls, axis=0)
    if len(u) != 4:
        return mask_color, None

    pts = get_points(ls)
    if any(x < 0 or y < 0 or x > frame.shape[1] or y > frame.shape[0] for x, y in pts):
        return mask_color, None

    for point in pts:
        cv2.circle(mask_color, point, 5, (0, 255, 0), thickness=-1)

    return mask_color, pts


def detect_field_from_color(frame):
    frame = normalize_comprehensive(frame)
    mask = cv2.inRange(frame, (78, 85, 70), (95, 95, 88))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = tuple(filter(lambda c: cv2.contourArea(c) > 2000, contours))

    mask_color = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(mask_color, contours, -1, (0, 255, 0), thickness=3)

    all_contours = np.concatenate(contours)
    rect = cv2.boundingRect(all_contours)

    return mask_color, rect
