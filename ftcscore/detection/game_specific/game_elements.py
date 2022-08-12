from dataclasses import dataclass
from enum import Enum, auto, unique
from typing import Iterable
import numpy as np
import numpy.typing as npt
import cv2
from ftcscore.processing.normalize import normalize_standard


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


# 46, 100, 100 : 31, 55, 84
# 23, 255, 255 : 15, 140, 214
def detect_cubes(overhead: BGRImage) -> Iterable[GameElement]:
    norm = normalize_standard(overhead)
    hsv = cv2.cvtColor(norm, cv2.COLOR_BGR2HSV)
    cv2.imshow('bruh', hsv)
    mask = cv2.inRange(hsv, (0, 50, 130), (40, 160, 255))
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    result = cv2.bitwise_and(norm, norm, mask=mask)
    cv2.imshow('cubes', cv2.resize(result, (1000, 1000)))
    return []


detectors = [detect_cubes]
