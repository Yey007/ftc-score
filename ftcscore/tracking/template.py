from ftcscore.processing.crop import crop_to_rect
import cv2
from ftcscore.tracking.types import Window, Image


class TrackerTemplate:
    track_window: Window

    template: Image
    w: int
    h: int

    method: int

    def __init__(self, method: int):
        self.method = method

    def init(self, frame, window: Window):
        self.track_window = window
        self.template = crop_to_rect(frame, window)
        self.w, self.h = self.template.shape[0], self.template.shape[1]

    def update(self, frame: Image):
        res = cv2.matchTemplate(frame, self.template, self.method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if self.method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            x, y = min_loc
        else:
            x, y = max_loc

        return True, (x, y, self.w, self.h)
