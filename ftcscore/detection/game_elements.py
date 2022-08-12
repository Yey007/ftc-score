from typing import Iterable, Tuple
from ftcscore.detection.game_specific.game_elements import GameElement, BGRImage, detectors
import itertools
import numpy as np
import cv2


def detect_all_elements(overhead: BGRImage) -> Tuple[BGRImage, Iterable[GameElement]]:
    all_elements = (detector(overhead) for detector in detectors)
    all_elements = itertools.chain.from_iterable(all_elements)
    all_elements = list(all_elements)

    overhead_copy = np.copy(overhead)
    for element in all_elements:
        cv2.rectangle(overhead_copy,
                      pt1=element.position,
                      pt2=element.position + element.size,
                      color=(0, 255, 0))

    return overhead_copy, all_elements
