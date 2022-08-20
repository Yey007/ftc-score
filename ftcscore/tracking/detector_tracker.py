import numpy as np


class DetectorTracker:
    def __init__(self, detector, tracker_factory, track_frequency):
        self.detector = detector
        self.tracker_factory = tracker_factory
        self.tracker = tracker_factory()
        self.tracking_started = False
        self.last_position = None
        self.frame_num = 0
        self.track_frequency = track_frequency

    def update(self, frame):
        if self.frame_num % self.track_frequency == 0:
            self.frame_num += 1
            bb = self.track_position(frame)
            bb = np.array(bb, dtype=np.uint16)
            self.last_position = bb
            return bb
        else:
            self.frame_num += 1
            return self.last_position

    def track_position(self, frame):
        if self.tracking_started:
            success, bb = self.tracker.update(frame)
            if success:
                return bb
            else:
                self.tracker = self.tracker_factory()
                bb = self.init_tracker(frame)
                return bb
        else:
            bb = self.init_tracker(frame)
            if bb is not None:
                self.tracking_started = True
            return bb

    def init_tracker(self, frame):
        bb = self.detector(frame)
        self.tracker.init(frame, bb)
        return bb
