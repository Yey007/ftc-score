from dataclasses import dataclass
from enum import Enum, auto, unique
from typing import Iterable
import numpy as np
import numpy.typing as npt
import cv2
from ftcscore.processing.crop import crop_to_rect
from ftcscore.processing.normalize import normalize_standard, normalize_lcn, normalize_comprehensive


@unique
class GameElementType(Enum):
    DUCK = auto()
    CUBE = auto()
    SPHERE = auto()


@dataclass
class GameElement:
    type: GameElementType
    position: npt.NDArray  # x, y
    size: npt.NDArray  # w, h


BGRImage = npt.NDArray[np.uint8]


#  Inspired by https://stackoverflow.com/questions/35226993/how-to-crop-away-convexity-defects
def best_defects(defects, contour):
    min_dist = 9999999
    start = None
    end = None

    med_defect_dist = np.median(defects, axis=(0, 1))[3]
    defect_dist_thresh = med_defect_dist - 100

    for i in range(defects.shape[0]):
        for j in range(i + 1, defects.shape[0]):
            p1_dist = defects[i, 0, 3]
            p2_dist = defects[j, 0, 3]
            p1i = defects[i, 0, 2]
            p2i = defects[j, 0, 2]
            p1i, p2i = max(p1i, p2i), min(p1i, p2i)
            p1 = tuple(contour[p1i][0])
            p2 = tuple(contour[p2i][0])
            dist_squared = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
            if min_dist > dist_squared and \
                    p1_dist > 175 and p2_dist > 175:
                start = p2i
                end = p1i
                min_dist = dist_squared

    return start, end


def split_cluster(cluster):
    hull = cv2.convexHull(cluster, returnPoints=False)
    try:
        defects = cv2.convexityDefects(cluster, hull)
    except Exception:
        return [cluster]  # Problem with opencv where convexity defects fails due to hull indices not being
        # monotonous but sorting doesn't help.
    if defects is None:
        return [cluster]

    first, last = best_defects(defects, cluster)
    if first is None or last is None:
        return [cluster]

    new_1 = cluster[first:last]
    new_1_split = split_cluster(new_1)
    new_2 = np.concatenate((cluster[:first], cluster[last:]), axis=0)
    new_2_split = split_cluster(new_2)
    return [*new_1_split, *new_2_split]


knn = cv2.ml.KNearest_create()
colors = np.array(
    [[160, 90, 70], [120, 100, 220], [160, 160, 160], [0, 0, 0], [255, 255, 255], [140, 200, 250],
     [100, 160, 212]]).astype(np.float32)
labels = np.arange(0, colors.shape[0])
knn.train(colors, cv2.ml.ROW_SAMPLE, labels)


def detect_cubes(overhead: BGRImage) -> Iterable[GameElement]:
    norm = normalize_standard(overhead)
    res = norm.reshape((-1, 3)).astype(np.float32)
    ret, result, neighbours, dist = knn.findNearest(res, k=1)
    result = result.astype(np.uint8)
    res = np.choose(result, colors)
    res = res.reshape(norm.shape).astype(np.uint8)

    mask = cv2.inRange(res, (140, 200, 250), (140, 200, 250))
    mask = cv2.bitwise_or(mask, cv2.inRange(res, (100, 160, 212), (100, 160, 212)))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)

    clusters = filter(lambda c: cv2.contourArea(c) >= 180, contours)
    individuals = list(filter(lambda c: 180 > cv2.contourArea(c) > 20, contours))

    for cluster in clusters:
        split = split_cluster(cluster)
        individuals.extend(split)

    individuals = filter(lambda c: cv2.contourArea(c) > 20, individuals)
    boxes = map(cv2.boundingRect, individuals)

    return [GameElement(type=GameElementType.CUBE, position=np.array((x, y)), size=np.array((w, h))) for x, y, w, h in
            boxes]


# def detect_cubes(overhead: BGRImage) -> Iterable[GameElement]:
#     norm = normalize_standard(overhead)
#     hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
#     cv2.imshow('bruh', hsv)
#     mask = cv2.inRange(hsv, (0, 30, 10), (30, 190, 255))
#     kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
#
#     result = cv2.bitwise_and(norm, norm, mask=mask)
#     cv2.imshow('cubes', result)
#     return []

# def detect_cubes(storage: BGRImage):
#     norm = normalize_lcn(storage)
#     norm = cv2.normalize(norm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
#     cv2.imshow('norm', norm)
#     mask = cv2.inRange(norm, (0, 0, 20), (35, 60, 130))
#
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
#
#     contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
#     clusters = filter(lambda c: cv2.contourArea(c) >= 200, contours)
#     individuals = filter(lambda c: 200 > cv2.contourArea(c) > 80, contours)
#
#     for cluster in clusters:
#         rect = cv2.boundingRect(cluster)
#         cv2.rectangle(storage, rect, color=(255, 0, 0))
#
#     for individual in individuals:
#         rect = cv2.boundingRect(individual)
#         cv2.rectangle(storage, rect, color=(0, 255, 0))
#
#     cv2.imshow('mask', mask)
#     return []


detectors = [detect_cubes]
