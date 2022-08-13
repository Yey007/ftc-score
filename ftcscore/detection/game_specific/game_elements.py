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


# def detect_cubes(overhead: BGRImage) -> Iterable[GameElement]:
#     scale = 2
#     overhead = cv2.resize(overhead, (int(overhead.shape[1] * scale), int(overhead.shape[0] * scale)))
#     norm = normalize_comprehensive(overhead)
#     hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
#     mask = cv2.inRange(hsv, (15, 30, 85), (40, 150, 110))
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
#
#     result = cv2.bitwise_and(overhead, overhead, mask=mask)
#
#     cv2.imshow('c', result)
#     return []

def detect_cubes(overhead: BGRImage) -> Iterable[GameElement]:
    norm = normalize_standard(overhead)
    hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
    cv2.imshow('bruh', hsv)
    mask = cv2.inRange(hsv, (5, 40, 120), (40, 190, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=2)

    result = cv2.bitwise_and(norm, norm, mask=mask)
    cv2.imshow('cubes', cv2.resize(result, (1000, 1000)))
    return []


detectors = [detect_cubes]
