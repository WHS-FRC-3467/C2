import cv2
import numpy
from config.config import ConfigStore
from vision_types import FiducialImageObservation


def overlay_image_observation(image: cv2.Mat, observation: FiducialImageObservation) -> None:
    cv2.aruco.drawDetectedMarkers(image, numpy.array([observation.corners]), numpy.array([observation.tag_id]))


