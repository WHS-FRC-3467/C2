import ctypes
import os
from typing import Any

import cv2
import numpy
from cv2.typing import MatLike

from config.config import ConfigStore
from pipeline.FiducialDetector import FiducialDetector
from vision_types import FiducialImageObservation

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

    def __init__(self) -> None:
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

    def create(self, dict_id: int) -> ctypes.c_void_p:
        return ctypes.c_void_p(self._lib.aruco_nano_create(dict_id))

    def destroy(self, handle: ctypes.c_void_p) -> None:
        self._lib.aruco_nano_destroy(handle)

    def detect(
        self,
        handle: ctypes.c_void_p,
        data_ptr: Any,
        width: int,
        height: int,
        stride: int,
        out_buf: Any,
        max_det: int,
    ) -> int:
        return self._lib.aruco_nano_detect(
            handle, data_ptr, width, height, stride, out_buf, max_det
        )


_lib = _ArucoNanoLib()


class ArucoNanoFiducialDetector(FiducialDetector):
    """AprilTag detector using the aruco_nano C++ library via ctypes."""

    def __init__(self, dictionary_id: int) -> None:
        self._handle: ctypes.c_void_p | None = _lib.create(dictionary_id)
        self._det_buf = (_Detection * MAX_DETECTIONS)()

    def __del__(self) -> None:
        if self._handle:
            _lib.destroy(self._handle)
            self._handle = None

    def detect_fiducials(
        self, image: MatLike, config_store: ConfigStore
    ) -> list[FiducialImageObservation]:
        image_array = numpy.asarray(image)
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array

        if not gray.flags["C_CONTIGUOUS"]:
            gray = numpy.ascontiguousarray(gray)

        height, width = gray.shape
        stride = gray.strides[0]
        data_ptr = gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        if self._handle is None:
            return []

        count = _lib.detect(
            self._handle, data_ptr, width, height, stride, self._det_buf, MAX_DETECTIONS
        )

        observations: list[FiducialImageObservation] = []
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
