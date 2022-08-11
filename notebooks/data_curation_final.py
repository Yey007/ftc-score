import os
import cv2
import pytesseract
import numpy as np
import ffmpeg

OUT_DIR = '../data/processed/videos/new'
IN_DIR = '../data/raw/videos'


def in_range(p, r):
    return r[0] + 10 >= p[0] >= r[0] - 10 and r[1] + 10 >= p[1] >= r[1] - 10 and r[2] + 10 >= p[2] >= r[2] - 10


def has_text(cropped, text):
    gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
    _, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return text == pytesseract.image_to_string(img_bin,
                                               config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789').strip()


def get_matches(video):
    cap = cv2.VideoCapture(f'{IN_DIR}/{video}')

    got_frame, _ = cap.read()
    if not got_frame:
        print(f'Skipping {video}')

    prev = (0, 0, 0)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = 0

    matches = []

    while True:
        got_frame, frame = cap.read()

        if not got_frame:
            break

        detection_range = frame[928:948, 739:741]
        text_range = frame[905:950, 910:1020]
        detection_range_avg = np.average(detection_range, axis=(0, 1))

        if in_range(prev, (150, 150, 150)) \
                and not in_range(detection_range_avg, (150, 150, 150)) \
                and has_text(text_range, "29"):
            # match start
            matches.append([frame_count - fps, frame_count + int((2.5 * 60 + 8) * fps)])
            print(f'Match started at {frame_count}')

        frame_count += 1
        prev = detection_range_avg
        print(f'{frame_count / total_frames * 100:.2f}%', end='\r')

    print(matches)

    return matches


def split_matches(video, matches):
    in_file = ffmpeg.input(f'{IN_DIR}/{video}')
    for (start, end) in matches:
        (
            in_file
            .trim(start_frame=start, end_frame=end)
            .setpts('PTS-STARTPTS')
            .output(f'{OUT_DIR}/match-{video}-{start}-{end}.mp4')
            .run()
        )


def process_video(video):
    matches = get_matches(video)
    split_matches(video, matches)


if __name__ == '__main__':
    for vid in os.listdir(IN_DIR):
        if os.path.isfile(f'{IN_DIR}/{vid}'):
            print(f'Processing {vid}')
            process_video(vid)
