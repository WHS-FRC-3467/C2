"""Debug drawing for the MJPEG stream."""

from __future__ import annotations

import cv2
import numpy


def draw_tag_detections(image: numpy.ndarray, detections) -> None:
    """Outline detected tags (anything with ``tag_id`` and (4, 2) ``corners``) on a BGR image."""
    if not detections:
        return
    corners = [d.corners.reshape(1, 4, 2).astype(numpy.float32) for d in detections]
    ids = numpy.array([[d.tag_id] for d in detections], dtype=numpy.int32)
    cv2.aruco.drawDetectedMarkers(image, corners, ids)
