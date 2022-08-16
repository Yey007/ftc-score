def rect_to_points(rect):
    x, y, w, h = rect
    return (x, y), (x + w, y), (x, y + h), (x + w, y + h)


def point_in_rect(rect, point):
    xr, yr, wr, hr = rect
    x, y = point
    return xr <= x <= xr + wr and yr <= y <= yr + hr


def rects_overlap(rect1, rect2):
    points = rect_to_points(rect2)
    return any(point_in_rect(rect1, p) for p in points)


def enlarge_rect(rect, pixels):
    x, y, w, h = rect
    return x - pixels, y - pixels, w + 2 * pixels, h + 2 * pixels
