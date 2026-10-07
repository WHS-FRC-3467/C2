from typing import List

import cv2
from config.Config import ConfigStore
from vision_types import FiducialImageObservation


class FiducialDetector:
    def __init__(self) -> None:
        raise NotImplementedError

    def detect_fiducials(
        self, image: cv2.Mat, config_store: ConfigStore
    ) -> List[FiducialImageObservation]:
        raise NotImplementedError
