import cv2
from ftcscore.detection.field import detect_field_edges
from ftcscore.processing.crop import crop_info_panel
from ftcscore.processing.perspective_transform import get_four_point_transform, apply_perspective_transform
from ftcscore.util.fps import get_fps, show_fps

video_source = cv2.VideoCapture('../data/processed/videos/new/match-wisconsin.mp4-119403-124173.mp4')
prev_frame_time = 0

got_frame, frame = video_source.read()
if not got_frame:
    raise Exception('Invalid video')

cropped = crop_info_panel(frame)
_, corners = detect_field_edges(cropped)
transform, out_shape = get_four_point_transform(corners)

while True:
    got_frame, frame = video_source.read()
    if not got_frame:
        break

    cropped = crop_info_panel(frame)
    overhead = apply_perspective_transform(cropped, transform, out_shape)

    fps, prev_frame_time = get_fps(prev_frame_time)
    show_fps(cropped, fps)

    cv2.imshow("video", cropped)
    cv2.imshow("overhead", overhead)

    if cv2.waitKey(30) == ord('q'):
        break
