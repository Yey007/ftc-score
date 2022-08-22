import cv2

from ftcscore.detection.field import detect_field_from_edges, detect_field_from_color
from ftcscore.detection.game_elements import detect_all_elements
from ftcscore.detection.game_specific.storage import detect_shared_hub
from ftcscore.processing.background_selection import McKennaBackgroundSubtractor
from ftcscore.processing.crop import crop_info_panel, crop_to_rect
from ftcscore.processing.normalize import normalize_standard
from ftcscore.processing.perspective_transform import get_four_point_transform
from ftcscore.tracking.detector_tracker import DetectorTracker
from ftcscore.util.fps import get_fps, show_fps
from ftcscore.util.rect import enlarge_rect

video_source = cv2.VideoCapture('../data/processed/videos/new/match-oregon.mp4-252003-256743.mp4')

width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(video_source.get(cv2.CAP_PROP_FPS))

prev_frame_time = 0
shared_tracker = DetectorTracker(detect_shared_hub, cv2.legacy.TrackerCSRT_create, track_frequency=3)

# Wait for field detection
while True:
    got_frame, frame = video_source.read()
    if not got_frame:
        break

    cropped = crop_info_panel(frame)
    field_detection, corners = detect_field_from_edges(cropped)

    cv2.imshow('field_detection', field_detection)
    cv2.waitKey(30)

    if corners is None:
        continue

    transform = get_four_point_transform(corners)
    overhead = transform(cropped)
    _, rect = detect_field_from_color(overhead)

    cv2.destroyAllWindows()
    break

while True:
    got_frame, frame = video_source.read()
    if not got_frame:
        break

    cropped = crop_info_panel(frame)
    overhead = transform(cropped)
    overhead = crop_to_rect(overhead, rect)

    r = shared_tracker.update(overhead)
    if r is not None:
        r = enlarge_rect(r, 10)
        cv2.rectangle(overhead, r, color=(0, 255, 0), thickness=2)

        norm = normalize_standard(overhead)
        cropped = crop_to_rect(norm, r)
        _, elements = detect_all_elements(norm)
        for element in elements:
            x, y = element.position
            x, y = x + r[0], y + r[1]
            w, h = element.size
            cv2.rectangle(overhead, (x, y), (x + w, y + h), color=(0, 0, 255), thickness=1)

    fps, prev_frame_time = get_fps(prev_frame_time)
    show_fps(overhead, fps)

    cv2.imshow("overhead", overhead)

    key = cv2.waitKey(30)
    if key == ord('q'):
        break
    elif key == ord(' '):
        key = cv2.waitKey(0)
        if key == ord('q'):
            break

cv2.destroyAllWindows()
video_source.release()

# Tracking - use the circular nature
# Assume robots deposit the cube correctly
# 4 Research Articles
# You can decide whether you want to talk about different stuff you tried
# First rough half (up to your work) (outline the whole thing)
# Channel variance is variance of channel average over time
# You got the separation algorithm, right?
