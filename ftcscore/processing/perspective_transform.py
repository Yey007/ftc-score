import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import cv2

# Modified from https://pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/


def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def get_four_point_transform(pts):
    points = order_points(pts)

    # the side length is the minimum distance between any of the 4 points.
    distances = euclidean_distances(points, points)  # Compute distances between all points

    # Get the smallest non-zero distance (0 means we compared the point with itself)
    flat = distances.flatten()
    side_length = flat[flat != 0].min()
    side_length = int(side_length)

    dst = np.array([
        [0, 0],
        [side_length - 1, 0],
        [side_length - 1, side_length - 1],
        [0, side_length - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(points, dst)

    return M, (side_length, side_length)


def apply_perspective_transform(image, M, shape):
    warped = cv2.warpPerspective(image, M, shape)
    return warped
