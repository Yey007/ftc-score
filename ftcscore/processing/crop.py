def crop_info_panel(frame):
    return frame[:825]


def crop_to_rect(frame, rect):
    x, y, w, h = rect
    return frame[y:y + h, x:x + w]
