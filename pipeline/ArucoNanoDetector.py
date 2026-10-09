import ctypes
import os
from typing import List, Optional

import cv2
import numpy
from input.Config import ConfigStore
from pipeline.VisionTypes import FiducialImageObservation

# Path to the shared library relative to this file
_LIB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "aruco_nano", "build", "libaruco_nano_capi.so"
)

MAX_DETECTIONS = 64


class _Detection(ctypes.Structure):
    _fields_ = [
        ("id", ctypes.c_int),
        ("corners", ctypes.c_float * 8),
    ]


class _ArucoNanoLib:
    """Thin ctypes wrapper around libaruco_nano_capi.so."""

    def __init__(self):
        self._lib = ctypes.cdll.LoadLibrary(_LIB_PATH)

        self._lib.aruco_nano_create.restype = ctypes.c_void_p
        self._lib.aruco_nano_create.argtypes = [ctypes.c_int]

        self._lib.aruco_nano_destroy.restype = None
        self._lib.aruco_nano_destroy.argtypes = [ctypes.c_void_p]

        self._lib.aruco_nano_detect.restype = ctypes.c_int
        self._lib.aruco_nano_detect.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(_Detection),
            ctypes.c_int,
        ]

    def create(self, dict_id: int):
        return self._lib.aruco_nano_create(dict_id)

    def destroy(self, handle):
        self._lib.aruco_nano_destroy(handle)

    def detect(self, handle, data_ptr, width, height, stride, out_buf, max_det):
        return self._lib.aruco_nano_detect(
            handle, data_ptr, width, height, stride, out_buf, max_det
        )


_lib = _ArucoNanoLib()


class ArucoNanoFiducialDetector:
    """AprilTag detector using the aruco_nano C++ library via ctypes."""

    ROI_PADDING_FRACTION = 1.0
    MIN_ROI_PADDING_PX = 24

    def __init__(self, dictionary_id: int) -> None:
        self._handle = _lib.create(dictionary_id)
        self._det_buf = (_Detection * MAX_DETECTIONS)()
        self._previous_observations: dict[int, numpy.ndarray] = {}
        self.last_crop_boxes: dict[tuple[int, ...], tuple[int, int, int, int]] = {}

    def __del__(self):
        if hasattr(self, "_handle") and self._handle:
            _lib.destroy(self._handle)
            self._handle = None

    def detect_fiducials(
        self,
        image: cv2.typing.MatLike,
        projected_tag_corners: Optional[dict[int, numpy.ndarray]] = None,
    ) -> List[FiducialImageObservation]:
        """Search projected tag ROIs and recover with a full-frame scan.

        Predicted corners come from projecting the known field layout using the
        selected robot pose. Previous detections identify tags to track for
        recovery. A full-frame scan runs when no usable crops exist, no tags are
        found in the crops, or a previously detected tag is missing.
        """
        height, width = image.shape[:2]
        self.last_crop_boxes = {}
        roi_by_id = {
            tag_id: corners
            for tag_id, corners in (projected_tag_corners or {}).items()
        }

        found_by_id: dict[int, FiducialImageObservation] = {}
        expected_ids: set[int] = set()
        boxes = []
        for tag_id, corners in roi_by_id.items():
            x1, y1, x2, y2 = self._padded_bounds(corners, width, height)
            if x2 <= x1 or y2 <= y1:
                continue
            expected_ids.add(tag_id)
            boxes.append(((tag_id,), (x1, y1, x2, y2)))

        self.last_crop_boxes = dict(self._merge_crop_boxes(boxes))
        for x1, y1, x2, y2 in self.last_crop_boxes.values():

            # Detection corners are crop-relative; restore full-frame
            # coordinates before handing observations to pose estimation.
            crop_observations = self._detect_full_frame(image[y1:y2, x1:x2])
            for observation in crop_observations:
                translated = numpy.asarray(observation.corners).copy()
                translated[..., 0] += x1
                translated[..., 1] += y1
                found_by_id[observation.tag_id] = FiducialImageObservation(
                    observation.tag_id, translated
                )

        # Recover with a full-image scan if there are no expected ids,
        # no tags were found, or a previously found tag is no longer there
        # Tags deliberately excluded by the nearest-four selection should
        # not force a full scan simply because their crops were not searched.
        previous_ids = set(self._previous_observations).intersection(expected_ids)
        missing_previous_ids = previous_ids - found_by_id.keys()

        should_scan_full = (
            not expected_ids
            or not found_by_id
            or bool(missing_previous_ids)
        )

        observations = (
            self._detect_full_frame(image)
            if should_scan_full
            else list(found_by_id.values())
        )

        self._previous_observations = {
            observation.tag_id: numpy.asarray(observation.corners)
            .reshape(4, 2)
            .copy()
            for observation in observations
        }
        return observations

    @staticmethod
    def _merge_crop_boxes(boxes):
        """Merge intersecting rectangles, including overlaps created by a merge."""
        merged = []
        for tag_ids, bounds in boxes:
            x1, y1, x2, y2 = bounds
            index = 0
            while index < len(merged):
                other_ids, (ox1, oy1, ox2, oy2) = merged[index]
                if x1 < ox2 and ox1 < x2 and y1 < oy2 and oy1 < y2:
                    tag_ids = tuple(sorted(set(tag_ids).union(other_ids)))
                    x1, y1 = min(x1, ox1), min(y1, oy1)
                    x2, y2 = max(x2, ox2), max(y2, oy2)
                    merged.pop(index)
                    index = 0
                else:
                    index += 1
            merged.append((tag_ids, (x1, y1, x2, y2)))
        return merged

    def _padded_bounds(
        self, corners: numpy.ndarray, image_width: int, image_height: int
    ) -> tuple[int, int, int, int]:
        points = numpy.asarray(corners).reshape(4, 2)
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)
        pad_x = max(
            self.MIN_ROI_PADDING_PX,
            (max_x - min_x) * self.ROI_PADDING_FRACTION,
        )
        pad_y = max(
            self.MIN_ROI_PADDING_PX,
            (max_y - min_y) * self.ROI_PADDING_FRACTION,
        )
        return (
            max(0, int(numpy.floor(min_x - pad_x))),
            max(0, int(numpy.floor(min_y - pad_y))),
            min(image_width, int(numpy.ceil(max_x + pad_x))),
            min(image_height, int(numpy.ceil(max_y + pad_y))),
        )

    def _detect_full_frame(
        self, image: cv2.typing.MatLike
    ) -> List[FiducialImageObservation]:
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        if not gray.flags["C_CONTIGUOUS"]:
            gray = numpy.ascontiguousarray(gray)

        height, width = gray.shape
        stride = gray.strides[0]
        data_ptr = gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        count = _lib.detect(
            self._handle, data_ptr, width, height, stride, self._det_buf, MAX_DETECTIONS
        )

        observations = []
        for i in range(count):
            det = self._det_buf[i]
            corners = numpy.array(
                [
                    [
                        [det.corners[0], det.corners[1]],
                        [det.corners[2], det.corners[3]],
                        [det.corners[4], det.corners[5]],
                        [det.corners[6], det.corners[7]],
                    ]
                ],
                dtype=numpy.float64,
            )
            observations.append(FiducialImageObservation(det.id, corners))

        return observations
