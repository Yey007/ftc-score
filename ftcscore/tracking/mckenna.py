import itertools
from collections import defaultdict
from dataclasses import dataclass
from enum import auto, Enum
from typing import Tuple, List
import cv2
import numpy as np
import numpy.typing as npt

from ftcscore.util.rect import rects_overlap


class TrackingStatus(Enum):
    Idk = auto()


@dataclass
class Region:
    contour: npt.NDArray[np.uint16]
    timestamp: int
    tracking_status: TrackingStatus = TrackingStatus.Idk

    @property
    def bounding_box(self):
        return cv2.boundingRect(self.contour)

    def __hash__(self):
        return id(self)


def regions_match(r1: Region, r2: Region) -> bool:
    return rects_overlap(r1.bounding_box, r2.bounding_box)


def full_join_lists(l1: List, l2: List, f) -> List:
    result = []
    l2_used = [False] * len(l2)

    for value in l1:
        for i, value2 in enumerate(l2):
            added = False
            if f(value, value2):
                l2_used[i] = True
                result.append((value, value2))
                added = True
            if not added:
                result.append((value, None))

    for i, value in enumerate(l2):
        if l2_used[i] is False:
            result.append((None, value))

    # Collapse
    d = defaultdict(list)
    for value1, value2 in result:
        if value2 is None:
            d[value1] = []
        else:
            d[value1].append(value2)

    result = list(d.items())
    single_value_used = [False] * len(result)
    result_copy = []
    for i, (value1, value2) in enumerate(result):
        if value1 is None:
            for v in value2:
                result_copy.append(([], [v]))
        elif len(value2) == 0:
            result_copy.append(([value1], []))
        elif len(value2) == 1:
            if not single_value_used[i]:
                single_value_used[i] = True
                matches = []
                for j, (inner_value1, inner_value2) in enumerate(result):
                    if not single_value_used[j] and len(inner_value2) == 1 and id(inner_value2[0]) == id(value2[0]):
                        matches.append(inner_value1)
                result_copy.append((matches, value2))
        else:
            result_copy.append(([value1], value2))

    return result_copy


class TrackerMcKenna:
    bg_subtractor: cv2.BackgroundSubtractor
    tracked_regions: List[Region]
    region_frame_thresh = 3

    def __init__(self, bg_subtractor, region_frame_thresh=3):
        self.bg_subtractor = bg_subtractor
        self.tracked_regions = []
        self.region_frame_thresh = region_frame_thresh

    def update(self, image):
        fg = self.bg_subtractor.apply(image)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(fg, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        regions = [Region(cnt, 0) for cnt in contours]

        corresponding_regions = full_join_lists(self.tracked_regions, regions, regions_match)

        new = 0
        delete = 0
        persist = 0
        split = 0
        merge = 0

        new_regions = []
        for tracked, detected in corresponding_regions:
            if len(tracked) == 0:  # New region
                assert len(detected) == 1
                new_regions.append(detected[0])
                new += 1
            elif len(tracked) == 1:
                if len(detected) == 0:  # Region deleted, nothing to do
                    delete += 1
                elif len(detected) == 1:  # Region persisted
                    new_regions.append(tracked[0])
                    persist += 1
                else:  # Region split
                    parent = tracked[0]

                    for region in detected:
                        region.timestamp = parent.timestamp
                        region.tracking_status = parent.tracking_status

                    new_regions.extend(detected)
                    split += 1
            else:  # Region merge
                assert len(detected) == 1
                merged = detected[0]
                oldest = max(tracked, key=lambda r: r.timestamp)
                merged.timestamp = oldest.timestamp
                merged.tracking_status = oldest.tracking_status
                new_regions.append(merged)
                merge += 1

        self.tracked_regions = new_regions
        print(persist)

        for region in new_regions:
            cv2.rectangle(image, region.bounding_box, color=(0, 255, 0))
