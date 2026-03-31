import cv2
import numpy
from cv2.typing import MatLike

from config.config import ConfigStore
from vision_types import FiducialImageObservation


class FiducialDetector:
    def __init__(self) -> None:
        raise NotImplementedError

    def detect_fiducials(
        self, image: MatLike, config_store: ConfigStore
    ) -> list[FiducialImageObservation]:
        raise NotImplementedError


class ArucoFiducialDetector(FiducialDetector):
    def __init__(self, dictionary_id: int) -> None:
        self._aruco_dict = cv2.aruco.getPredefinedDictionary(dictionary_id)
        self._aruco_params = cv2.aruco.DetectorParameters()
        self._detector = cv2.aruco.ArucoDetector(self._aruco_dict, self._aruco_params)

    def detect_fiducials(
        self, image: MatLike, config_store: ConfigStore
    ) -> list[FiducialImageObservation]:
        corners, ids, _ = self._detector.detectMarkers(image)
        if len(corners) == 0 or ids is None:
            return []
        observations: list[FiducialImageObservation] = []
        for tag_id, corner in zip(ids, corners):
            observations.append(
                FiducialImageObservation(
                    int(tag_id[0]),
                    numpy.asarray(corner, dtype=numpy.float64),
                )
            )
        return observations
