import cv2
from ftcscore.detection.field import detect_field_from_edges, detect_field_from_color
from ftcscore.detection.game_elements import detect_all_elements
from ftcscore.detection.game_specific.storage import detect_alliance_hub_blue, detect_alliance_hub_red, \
    detect_shared_hub
from ftcscore.processing.background_selection import McKennaBackgroundSubtractor
from ftcscore.processing.crop import crop_info_panel, crop_to_rect
from ftcscore.processing.perspective_transform import get_four_point_transform
from ftcscore.tracking.mckenna import TrackerMcKenna
from ftcscore.util.fps import get_fps, show_fps

video_source = cv2.VideoCapture('../data/processed/videos/new/match-oregon.mp4-252003-256743.mp4')

width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))

frame_num = 0
prev_frame_time = 0

bg_subtractor = McKennaBackgroundSubtractor()

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

    bg = bg_subtractor.apply(overhead)
    cv2.imshow('win', bg)

    fps, prev_frame_time = get_fps(prev_frame_time)
    show_fps(cropped, fps)

    cv2.imshow("video", cropped)
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
