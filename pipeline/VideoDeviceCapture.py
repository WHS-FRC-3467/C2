from typing import Optional, Tuple, Union

import cv2
import numpy


class VideoDeviceCapture:
    def __init__(self, device: Union[str, int]) -> None:
        self._device = device
        self._video: Optional[cv2.VideoCapture] = None

    def read(self) -> Tuple[bool, cv2.Mat]:
        if self._video is None:
            self._video = cv2.VideoCapture(self._device, cv2.CAP_V4L2)
            if not self._video.isOpened():
                self._video.release()
                self._video = None
                return False, numpy.array([])

        success, frame = self._video.read()
        if not success:
            self._video.release()
            self._video = None
            return False, numpy.array([])
        return success, frame

    def release(self) -> None:
        if self._video is not None:
            self._video.release()
            self._video = None
