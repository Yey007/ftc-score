import cv2
import numpy as np
from scipy import ndimage
from sklearn.metrics import mean_squared_error
from scipy.signal import convolve


# From paper: Comprehensive Colour Image Normalization
def normalize_comprehensive(frame):
    num_pixels = frame.shape[0] * frame.shape[1]
    frame = frame.astype('float32') / 255

    previous = frame
    delta = 1

    while delta > 0.001:
        sums = np.sum(frame, axis=2, keepdims=True)
        sums[sums == 0] = 1
        frame = frame / sums
        sums = np.sum(frame, axis=(0, 1))
        sums[sums == 0] = 1
        frame = (num_pixels / 3) * frame / sums

        delta = mean_squared_error(previous.flatten(), frame.flatten())
        previous = frame

    frame *= 255
    return cv2.convertScaleAbs(frame)


# Normalizing just pixels but same algorithm as above
def normalize_pixels(frame):
    sums = np.sum(frame, axis=2, keepdims=True)
    sums[sums == 0] = 1
    frame = frame / sums
    frame *= 255
    return frame.astype('uint8')


# From paper: Adaptive Local Contrast Normalization
def normalize_standard(frame):
    std = (frame - np.mean(frame)) / np.std(frame)
    return std


# From paper: Adaptive Local Contrast Normalization
# https://stackoverflow.com/questions/67056641/how-to-generate-2d-gaussian-kernel-using-2d-convolution-in-python
# was too slow, this worked https://stackoverflow.com/questions/16719720/is-convolution-slower-in-numpy-than-in-matlab
# still kinda slow, i'm also not confident I implemented this right
def normalize_lcn(frame):
    oned_kernel = cv2.getGaussianKernel(5, 2)
    kernel = convolve(oned_kernel, oned_kernel.T)
    intensities = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    convolved_grey = ndimage.convolve(intensities, kernel)
    return cv2.subtract(frame, cv2.cvtColor(convolved_grey, cv2.COLOR_GRAY2BGR))


def normalize_min_max(frame):
    return cv2.normalize(frame, None,
                         alpha=0, beta=255,
                         norm_type=cv2.NORM_MINMAX,
                         dtype=cv2.CV_8U)
