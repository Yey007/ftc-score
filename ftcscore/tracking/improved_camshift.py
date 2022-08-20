from typing import Tuple
import numpy as np
import numpy.typing as npt
import cv2
from ftcscore.processing.crop import crop_to_rect

Image = npt.NDArray[np.uint8]
Window = Tuple[np.float32, np.float32, np.float32, np.float32]

TERM_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)


def compute_histogram(frame: Image, window: Window):
    roi = crop_to_rect(frame, window)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, (0, 60, 32), (180., 255., 255.))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    return roi_hist


class TrackerImprovedCamshift:
    track_window: Window
    hist: Image
    akaze: cv2.AKAZE

    def __init__(self):
        self.akaze = cv2.AKAZE_create()

    def init(self, frame, window: Window):
        self.track_window = window
        self.hist = compute_histogram(frame, window)

    def update(self, frame: Image):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)
        _, track_window = cv2.meanShift(dst, self.track_window, TERM_CRIT)
        return True, track_window

    def is_occluded(self, frame: Image, occ_thresh: float):
        hist_object = self.hist
        hist_scene = compute_histogram(frame, self.track_window)
        bc = np.sum(hist_scene * hist_object)

        obj_mean = np.mean(hist_object)
        sce_mean = np.mean(hist_scene)

        n = ...  # Num points matched by akaze

        return np.sqrt(1 - (1 / (obj_mean * sce_mean * n ** 2)) * bc)
