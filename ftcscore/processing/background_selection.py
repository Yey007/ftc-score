import numpy
import numpy as np
import numpy.typing as npt

import cv2


class McKennaBackgroundSubtractor(cv2.BackgroundSubtractor):
    prev_pixel_mean: npt.NDArray[np.float64] | None
    prev_pixel_var: npt.NDArray[np.float64] | None
    prev_cam_mean: npt.NDArray[np.float] | None
    prev_cam_var: npt.NDArray[np.float] | None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_pixel_mean = None
        self.prev_pixel_var = None
        self.prev_cam_mean = None
        self.prev_cam_var = None

    def apply(self, image, fg_mask=None, learning_rate=0.9):
        alpha = 1 - learning_rate  # Smaller alpha -> learn faster (opposite of OpenCV spec)

        if self.prev_pixel_mean is None or self.prev_pixel_var is None:
            self.prev_pixel_mean = image
            self.prev_pixel_var = np.zeros(image.shape)
            self.prev_cam_mean = np.mean(image, axis=(0, 1))
            self.prev_cam_var = np.zeros(image.shape[2])

        pixel_mean = alpha * self.prev_pixel_mean + (1 - alpha) * image
        pixel_var = alpha * (self.prev_pixel_var + (pixel_mean - self.prev_pixel_mean) ** 2) \
                    + (1 - alpha) * (image - pixel_mean) ** 2
        cam_mean = alpha * self.prev_cam_mean + (1 - alpha) * image
        cam_var = alpha * (self.prev_cam_var + (cam_mean - self.prev_cam_mean) ** 2) \
                  + (1 - alpha) * (image - cam_mean) ** 2

        mask = np.abs(image - pixel_mean) > (3 * np.maximum(cam_var, pixel_var))

        self.prev_pixel_mean = pixel_mean
        self.prev_pixel_var = pixel_var
        self.prev_cam_mean = cam_mean
        self.prev_cam_var = cam_var

        return np.where(np.max(mask, axis=2), 255, 0).astype(np.uint8)

    def getBackgroundImage(self, background_image=None):
        pass
