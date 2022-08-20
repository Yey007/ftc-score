import math
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np


def lines_to_slopes(lines):
    delta_y = lines[:, 0, 1] - lines[:, 0, 3]
    delta_x = lines[:, 0, 0] - lines[:, 0, 2]
    return delta_y / delta_x


def lines_to_lengths(lines):
    return np.sqrt(
        np.square(lines[:, 0, 0] - lines[:, 0, 2]) + np.square(lines[:, 0, 1] - lines[:, 0, 3])
    )


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
        raise Exception('Parallel lines do not intersect. Did you pass the same line?')
    return int(x / z), int(y / z)


@dataclass(eq=True, frozen=True)
class Line:
    start: Tuple[np.float32, np.float32]
    end: Tuple[np.float32, np.float32]

    @property
    def len(self):
        x1, y1 = self.start
        x2, y2 = self.end
        return np.sqrt(np.square(x2 - x1) + np.square(y2 - y1))

    @property
    def inc(self):
        x1, y1 = self.start
        x2, y2 = self.end
        return np.arctan2(y2 - y1, x2 - x1)


# Algorithm 2
def merge_two_lines(l1, l2, dist_thresh, inc_thresh):
    if l1.len < l2.len:
        l1, l2 = l2, l1

    all_endpoint_pairs = [(l1.start, l2.start), (l1.start, l2.end), (l1.end, l2.start), (l1.end, l2.end)]
    endpoint_lines = [Line(p1, p2) for p1, p2 in all_endpoint_pairs]
    min_dist = min(endpoint_lines, key=lambda l: l.len)
    d = min_dist.len

    tao_s = dist_thresh * l1.len
    if d > tao_s:
        m = None
        return m

    l2_hat = l2.len / l1.len  # Eq 5
    d_hat = d / tao_s  # Eq 6
    lamb = l2_hat + d_hat  # Eq 7
    tao_theta = (1 - 1 / (1 + np.exp(-2 * (lamb - 1.5)))) * inc_thresh  # Eq 8

    theta_diff = np.abs(l2.inc - l1.inc)
    if theta_diff < tao_theta or theta_diff > (np.pi - tao_theta):
        m = max(endpoint_lines, key=lambda l: l.len)
        theta_m = m.inc
        if np.abs(l1.inc - theta_m) > 0.5 * inc_thresh:
            m = None
    else:
        m = None
        return m

    return m


# Algorithm 1
def merge_lines(lines, dist_thresh, inc_thresh):
    lines: List[Line] = [Line((x1, y1), (x2, y2)) for x1, y1, x2, y2 in lines[:, 0]]
    while True:
        n = len(lines)
        lines.sort(key=lambda l: l.len)

        removals = set()
        replacements = dict()

        for l1 in lines:
            if l1 in removals:
                continue
            if l1 in replacements:
                l1 = replacements[l1]

            with_remove_replace = [replacements.get(l, l) for l in lines if l not in removals]
            with_remove_replace.remove(l1)

            tao_s = dist_thresh * l1.len
            p = filter(lambda l2: np.abs(l2.inc - l1.inc) < inc_thresh, with_remove_replace)  # Eq 1
            p = filter(lambda l2:  # Eq 2
                       np.abs(l1.start[0] - l2.start[0]) < tao_s or
                       np.abs(l1.start[0] - l2.end[0]) < tao_s or
                       np.abs(l1.end[0] - l2.start[0]) < tao_s or
                       np.abs(l1.end[0] - l2.end[0]) < tao_s,
                       p)
            p = filter(lambda l2:  # Eq 3
                       np.abs(l1.start[1] - l2.start[1]) < tao_s or
                       np.abs(l1.start[1] - l2.end[1]) < tao_s or
                       np.abs(l1.end[1] - l2.start[1]) < tao_s or
                       np.abs(l1.end[1] - l2.end[1]) < tao_s,
                       p)

            r = set()

            for l2 in p:
                if l2 in r or l2 in removals:
                    continue

                if l2 in replacements:
                    l2 = replacements[l2]

                m = merge_two_lines(l1, l2, dist_thresh, inc_thresh)
                if m is not None:
                    replacements[l1] = m
                    r.add(l2)

            removals = removals.union(r)

        lines = [replacements.get(l, l) for l in lines if l not in removals]

        if n == len(lines):
            break

    return np.array([[[*line.start, *line.end]] for line in lines])
