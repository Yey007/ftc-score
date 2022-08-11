import time

import cv2


# https://www.geeksforgeeks.org/python-displaying-real-time-fps-at-which-webcam-video-file-is-processed-using-opencv/
def get_fps(prev_time):
    t = time.perf_counter()
    fps = 1 / (t - prev_time)
    return fps, t


def show_fps(frame, fps):
    cv2.putText(frame, f'FPS: {fps:.0f}', org=(50, 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=(255, 0, 0), thickness=3)
