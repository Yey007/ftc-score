from dataclasses import dataclass
from enum import Enum, auto, unique
from typing import Iterable
import numpy as np
import numpy.typing as npt
import cv2

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


def detect_cubes(overhead: BGRImage) -> Iterable[GameElement]:
    norm = normalize_comprehensive(overhead)
    hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20, 30, 85), (40, 150, 110))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imshow('m', mask)

    contours, _ = cv2.findContours(mask, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    clusters = filter(lambda c: cv2.contourArea(c) >= 450, contours)
    individuals = filter(lambda c: 450 > cv2.contourArea(c) > 130, contours)

    for cluster in clusters:
        rect = cv2.boundingRect(cluster)
        cv2.rectangle(overhead, rect, color=(0, 255, 0))

    for individual in individuals:
        rect = cv2.boundingRect(individual)
        cv2.rectangle(overhead, rect, color=(255, 0, 0))

    return []

# def detect_cubes(overhead: BGRImage) -> Iterable[GameElement]:
#     norm = normalize_standard(overhead)
#     hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
#     cv2.imshow('bruh', hsv)
#     mask = cv2.inRange(hsv, (0, 40, 120), (30, 190, 255))
#     # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
#     # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
#     # kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
#     # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)
#
#     result = cv2.bitwise_and(norm, norm, mask=mask)
#     cv2.imshow('cubes', cv2.resize(result, (1000, 1000)))
#     return []


detectors = [detect_cubes]
