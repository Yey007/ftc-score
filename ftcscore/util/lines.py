import numpy as np


def lines_to_distances(lines):
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
        raise Exception('Parallel lines do not intersect')
    return int(x / z), int(y / z)
