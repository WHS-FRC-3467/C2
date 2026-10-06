import cv2
import numpy
from vision_types import FiducialImageObservation


def overlay_image_observation(
    image: cv2.typing.MatLike, observation: FiducialImageObservation
) -> None:
    corners = [numpy.asarray(observation.corners, dtype=numpy.float32)]
    ids = numpy.asarray([[observation.tag_id]], dtype=numpy.int32)
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
