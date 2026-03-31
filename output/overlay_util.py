import cv2
import numpy
from cv2.typing import MatLike

from vision_types import FiducialImageObservation, ObjectDetectionObservation


def overlay_image_observation(
    image: MatLike, observation: FiducialImageObservation
) -> None:
    _ = cv2.aruco.drawDetectedMarkers(
        image,
        [observation.corners],
        numpy.array([[observation.tag_id]], dtype=numpy.int32),
    )


def overlay_object_detection(
    image: MatLike, detection: ObjectDetectionObservation
) -> None:
    cv2.rectangle(
        image,
        (detection.x0, detection.y0),
        (detection.x1, detection.y1),
        (0, 255, 0),
        2,
    )
    center = (int(round(detection.centroid_x)), int(round(detection.centroid_y)))
    cv2.circle(image, center, 3, (0, 0, 255), -1)
    label = (
        f"id={detection.class_id} conf={detection.confidence:.2f} "
        f"yaw={detection.yaw_deg:.1f} pitch={detection.pitch_deg:.1f} area={detection.area_px}"
    )
    text_origin = (detection.x0, max(18, detection.y0 - 8))
    cv2.putText(
        image,
        label,
        text_origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
