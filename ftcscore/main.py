import cv2

from ftcscore.detection.field import detect_field_from_edges, detect_field_from_color
from ftcscore.detection.game_specific.storage import detect_shared_hub
from ftcscore.processing.crop import crop_info_panel, crop_to_rect
from ftcscore.processing.normalize import normalize_standard, normalize_comprehensive
from ftcscore.processing.perspective_transform import get_four_point_transform
from ftcscore.util.fps import get_fps, show_fps

video_source = cv2.VideoCapture('../data/processed/videos/new/match-wisconsin.mp4-216195-220965.mp4')

width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_fps = int(video_source.get(cv2.CAP_PROP_FPS))

frame_num = 0
prev_frame_time = 0

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

    overhead = normalize_standard(overhead)
    shared = detect_shared_hub(overhead)
    if shared is not None:
        cv2.circle(overhead, (shared[0], shared[1]), shared[2], color=(0, 255, 0), thickness=2)

    # blue_hub = detect_alliance_hub_blue(overhead)
    # red_hub = detect_alliance_hub_red(overhead)
    #
    # for hub in (blue_hub, red_hub, shared_hub):
    #     cv2.rectangle(overhead, hub, color=(0, 255, 0), thickness=2)

    # shared_hub = detect_shared_hub(overhead)
    # only_shared = crop_to_rect(overhead, shared_hub)
    # img, boxes = detect_all_elements(only_shared)
    # print(len(boxes))

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

    frame_num += 1

cv2.destroyAllWindows()
video_source.release()

# Tracking - use the circular nature
# Assume robots deposit the cube correctly
# 4 Research Articles
# You can decide whether you want to talk about different stuff you tried
# First rough half (up to your work) (outline the whole thing)
# Channel variance is variance of channel average over time
# You got the separation algorithm, right?
