"""AprilTag detection with the aruco_nano C++ library (aruco_nano/), through ctypes.

Build the library first, from the repo root:
    cmake -S src/pipelines/apriltag/aruco_nano -B src/pipelines/apriltag/aruco_nano/build
    cmake --build src/pipelines/apriltag/aruco_nano/build
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy

_LIB_PATH = Path(__file__).resolve().parent / "aruco_nano" / "build" / "libaruco_nano_capi.so"
_MAX_DETECTIONS = 64


@dataclass(frozen=True)
class TagDetection:
    tag_id: int
    corners: numpy.ndarray
    """Pixel corners, shape (4, 2), float64, in aruco_nano's order."""


class _CDetection(ctypes.Structure):
    _fields_ = [("id", ctypes.c_int), ("corners", ctypes.c_float * 8)]


def _load_library() -> ctypes.CDLL:
    lib = ctypes.cdll.LoadLibrary(str(_LIB_PATH))
    lib.aruco_nano_create.restype = ctypes.c_void_p
    lib.aruco_nano_create.argtypes = [ctypes.c_int]
    lib.aruco_nano_destroy.restype = None
    lib.aruco_nano_destroy.argtypes = [ctypes.c_void_p]
    lib.aruco_nano_detect.restype = ctypes.c_int
    lib.aruco_nano_detect.argtypes = [
        ctypes.c_void_p,  # detector
        ctypes.POINTER(ctypes.c_uint8),  # image data
        ctypes.c_int,  # width
        ctypes.c_int,  # height
        ctypes.c_int,  # row stride, bytes
        ctypes.POINTER(_CDetection),  # output buffer
        ctypes.c_int,  # output buffer size
    ]
    return lib


_lib: ctypes.CDLL | None = None


class ArucoNanoDetector:
    """Not thread-safe: use one instance per thread."""

    def __init__(self, dictionary_id: int = cv2.aruco.DICT_APRILTAG_36h11) -> None:
        global _lib
        if _lib is None:
            _lib = _load_library()
        self._lib = _lib
        self._handle = _lib.aruco_nano_create(dictionary_id)
        self._results = (_CDetection * _MAX_DETECTIONS)()

    def close(self) -> None:
        if self._handle:
            self._lib.aruco_nano_destroy(self._handle)
            self._handle = None

    def __del__(self) -> None:
        self.close()

    def detect(self, image: numpy.ndarray) -> list[TagDetection]:
        """Detect tags in a grayscale (or BGR) image. Row-strided views are fine; no copy is made."""
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image.dtype != numpy.uint8 or image.strides[1] != 1:
            image = numpy.ascontiguousarray(image, dtype=numpy.uint8)

        height, width = image.shape
        count = self._lib.aruco_nano_detect(
            self._handle,
            image.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            width,
            height,
            image.strides[0],
            self._results,
            _MAX_DETECTIONS,
        )
        return [
            TagDetection(r.id, numpy.array(r.corners, dtype=numpy.float64).reshape(4, 2))
            for r in self._results[:count]
        ]
