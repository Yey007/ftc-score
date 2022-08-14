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


def regions_correspond(r1: Region, r2: Region) -> bool:
    return rects_overlap(r1.bounding_box, r2.bounding_box)


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
        contours = cv2.findContours(fg, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
        regions = [Region(cnt, 0) for cnt in contours]

        tracked_with_correspondence = []
        for region in self.tracked_regions:
            corresponding_regions = filter(lambda r: regions_correspond(region, r), regions)
            corresponding_regions = list(corresponding_regions)
            tracked_with_correspondence.append((region, corresponding_regions))

        new_tracked = []
        for region, correspondence in tracked_with_correspondence:
            if len(correspondence) == 0:  # Region deleted
                continue
            elif len(correspondence) == 1:  # Region persisted
                new_tracked.append(region)
            elif len(correspondence) > 1:  # Region split
                pass