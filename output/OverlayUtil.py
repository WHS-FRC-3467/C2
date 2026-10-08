import cv2
import numpy
from pipeline.VisionTypes import FiducialImageObservation


def overlay_crop_boxes(
    image: cv2.typing.MatLike,
    crop_boxes: dict[int, tuple[int, int, int, int]],
) -> None:
    """Draw the exact crop bounds searched by a camera's detector."""
    for tag_id, (x1, y1, x2, y2) in crop_boxes.items():
        # Crop end coordinates are exclusive; draw inside the searched pixels.
        cv2.rectangle(image, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 0), 1)
        cv2.putText(
            image,
            f"ROI {tag_id}",
            (x1 + 2, min(y1 + 15, image.shape[0] - 1)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )


def overlay_image_observation(
    image: cv2.typing.MatLike, observation: FiducialImageObservation
) -> None:
    corners = [numpy.asarray(observation.corners, dtype=numpy.float32)]
    ids = numpy.asarray([[observation.tag_id]], dtype=numpy.int32)
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
