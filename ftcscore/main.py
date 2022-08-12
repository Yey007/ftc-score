import cv2
from ftcscore.detection.field import detect_field_from_edges, detect_field_from_color
from ftcscore.detection.game_elements import detect_all_elements
from ftcscore.processing.crop import crop_info_panel, crop_to_rect
from ftcscore.processing.perspective_transform import get_four_point_transform
from ftcscore.util.fps import get_fps, show_fps

video_source = cv2.VideoCapture('../data/processed/videos/new/match-oregon.mp4-252003-256743.mp4')
tracker = cv2.TrackerMIL_create()

width = int(video_source.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video_source.get(cv2.CAP_PROP_FRAME_HEIGHT))

prev_frame_time = 0

initBB = None

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
    # detections, _ = detect_all_elements(overhead)

    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(overhead)
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(overhead, (x, y), (x + w, y + h),
                          (0, 255, 0), 2)

    fps, prev_frame_time = get_fps(prev_frame_time)
    show_fps(cropped, fps)

    cv2.imshow("video", cropped)
    cv2.imshow("overhead", overhead)
    # cv2.imshow('detections', detections)

    key = cv2.waitKey(30)
    if key == ord('q'):
        break
    elif key == ord(' '):
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
    elif key == ord('s'):
        initBB = cv2.selectROI("overhead", overhead, fromCenter=False,
                               showCrosshair=True)
        tracker.init(frame, initBB)

cv2.destroyAllWindows()
video_source.release()
