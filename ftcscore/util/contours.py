import cv2


def contour_y(c):
    moments = cv2.moments(c)
    if moments['m00'] != 0:
        return moments['m01'] / moments['m00']
    else:
        return float('inf')


def contour_x(c):
    moments = cv2.moments(c)
    if moments['m00'] != 0:
        return moments['m10'] / moments['m00']
    else:
        return float('inf')


def contour_aspect_ratio(c):
    x, y, w, h = cv2.boundingRect(c)
    aspect_ratio = float(w) / h
    return aspect_ratio