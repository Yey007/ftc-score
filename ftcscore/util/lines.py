import math
from typing import List, Iterable
import numpy as np

Line = np.ndarray


def merge_line_segments(line_i: Line, line_j: Line) -> Line:
    # line distance
    line_i_length = math.hypot(line_i[2] - line_i[0], line_i[3] - line_i[1])
    line_j_length = math.hypot(line_j[2] - line_j[0], line_j[3] - line_j[1])

    # centroids
    Xg = line_i_length * (line_i[0] + line_i[2]) + line_j_length * (line_j[0] + line_j[2])
    Xg /= 2 * (line_i_length + line_j_length)

    Yg = line_i_length * (line_i[1] + line_i[3]) + line_j_length * (line_j[1] + line_j[3])
    Yg /= 2 * (line_i_length + line_j_length)

    # orientation
    orientation_i = math.atan2((line_i[1] - line_i[3]), (line_i[0] - line_i[2]))
    orientation_j = math.atan2((line_j[1] - line_j[3]), (line_j[0] - line_j[2]))
    orientation_r = math.pi
    if (abs(orientation_i - orientation_j) <= math.pi / 2):
        orientation_r = line_i_length * orientation_i + line_j_length * orientation_j
        orientation_r /= line_i_length + line_j_length
    else:
        orientation_r = line_i_length * orientation_i + line_j_length * (
                orientation_j - math.pi * orientation_j / abs(orientation_j))
        orientation_r /= line_i_length + line_j_length

    # coordinate transformation
    # δXG = (δy - yG)sinθr + (δx - xG)cosθr
    # δYG = (δy - yG)cosθr - (δx - xG)sinθr
    a_x_g = (line_i[1] - Yg) * math.sin(orientation_r) + (line_i[0] - Xg) * math.cos(orientation_r)
    a_y_g = (line_i[1] - Yg) * math.cos(orientation_r) - (line_i[0] - Xg) * math.sin(orientation_r)

    b_x_g = (line_i[3] - Yg) * math.sin(orientation_r) + (line_i[2] - Xg) * math.cos(orientation_r)
    b_y_g = (line_i[3] - Yg) * math.cos(orientation_r) - (line_i[2] - Xg) * math.sin(orientation_r)

    c_x_g = (line_j[1] - Yg) * math.sin(orientation_r) + (line_j[0] - Xg) * math.cos(orientation_r)
    c_y_g = (line_j[1] - Yg) * math.cos(orientation_r) - (line_j[0] - Xg) * math.sin(orientation_r)

    d_x_g = (line_j[3] - Yg) * math.sin(orientation_r) + (line_j[2] - Xg) * math.cos(orientation_r)
    d_y_g = (line_j[3] - Yg) * math.cos(orientation_r) - (line_j[2] - Xg) * math.sin(orientation_r)

    # line distance relative
    line_i_rel_length = math.hypot(b_x_g - a_x_g, b_y_g - a_y_g)
    line_j_rel_length = math.hypot(d_x_g - c_x_g, d_y_g - c_y_g)

    # orthogonal projections over the axis X
    start_f = min(a_x_g, b_x_g, c_x_g, d_x_g)
    end_f = max(a_x_g, b_x_g, c_x_g, d_x_g)
    length_f = math.hypot(end_f - start_f, 0 - 0)

    # start_f = line_i_rel_length * math.cos(orientation_r)
    # end_f = line_j_rel_length * math.cos(orientation_r)

    start_x = int(Xg - start_f * math.cos(orientation_r))
    start_y = int(Yg - start_f * math.sin(orientation_r))
    end_x = int(Xg - end_f * math.cos(orientation_r))
    end_y = int(Yg - end_f * math.sin(orientation_r))

    return np.array((start_x, start_y, end_x, end_y))


class LineGroup:
    lines: List[Line]
    grouped_line: Line

    def __init__(self, line: Line):
        self.lines = [line]
        self.grouped_line = line

    def add_line(self, line: Line):
        self.lines.append(line)
        self.grouped_line = merge_line_segments(self.grouped_line, line)

    def fits_group(self, line: Line, inc_tolerance: float, dist_tolerance: float):
        # https://stackoverflow.com/a/48137604
        p1 = line[:2]
        p2 = line[2:]

        lp1 = self.grouped_line[:2]
        lp2 = self.grouped_line[2:]

        # Yes this isn't really the distance between the two segments,
        # but it's good enough as a heuristic
        p1p2 = np.array([p1, p2])
        dists = np.cross(lp2 - lp1, p1p2 - lp1) / np.linalg.norm(lp2 - lp1)
        dist_min = np.min(dists)

        x1, y1 = p1
        x2, y2 = p2
        lx1, ly1 = lp1
        lx2, ly2 = lp2
        la = np.rad2deg(np.arctan2(y2 - y1, x2 - x1))
        gla = np.rad2deg(np.arctan2(ly2 - ly1, ly2 - ly1))

        return dist_min < dist_tolerance and abs(gla - la) < inc_tolerance


def merge_lines(lines, incl_tolerance: float, dist_tolerance: float):
    groups: List[LineGroup] = []
    for line in lines:
        line = line[0]
        matched = False
        for group in groups:
            if group.fits_group(line, incl_tolerance, dist_tolerance):
                group.add_line(line)
        if not matched:
            groups.append(LineGroup(line))
    return np.array([[group.grouped_line] for group in groups]).astype('float32')


def lines_to_distances(lines):
    return np.sqrt(np.square(lines[:, 0, 0] - lines[:, 0, 2]) + np.square(lines[:, 0, 1] - lines[:, 0, 3]))


# https://stackoverflow.com/a/42727584
def intersection(l1, l2):
    a1, a2 = l1[0][:2], l1[0][2:]
    b1, b2 = l2[0][:2], l2[0][2:]

    s = np.vstack([a1, a2, b1, b2])  # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])  # get first line
    l2 = np.cross(h[2], h[3])  # get second line
    x, y, z = np.cross(l1, l2)  # point of intersection
    if z == 0:  # lines are parallel
        raise Exception('Parallel lines do not intersect')
    return int(x / z), int(y / z)
