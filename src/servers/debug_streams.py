"""The camera streams: every camera with pipeline overlays, and each camera raw.

The dashboard (servers/dashboard_server.py, on web_port) serves all of them.
Raw streams have no overlay: handy for focusing and calibration.
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import cv2
import numpy

from core.frames import FrameSet
from pipelines.pipeline import Overlay
from servers.mjpeg_server import MjpegStream

ANNOTATED_JPEG_QUALITY = 80
RAW_JPEG_QUALITY = 95


class DebugStreams:
    def __init__(self, num_cameras: int, overlays: Callable[[], list[Overlay]]) -> None:
        self._overlays = overlays
        self._lock = threading.Lock()
        self._frames: FrameSet | None = None
        self._annotated_cache: tuple[int, numpy.ndarray] | None = None

        self.annotated = MjpegStream(self._annotated_combined, ANNOTATED_JPEG_QUALITY)
        """Every camera side by side, with overlays."""
        self.annotated_cameras = [
            MjpegStream(self._annotated_source(i), ANNOTATED_JPEG_QUALITY)
            for i in range(num_cameras)
        ]
        self.raw_cameras = [
            MjpegStream(self._raw_source(i), RAW_JPEG_QUALITY)
            for i in range(num_cameras)
        ]

    def set_frames(self, frames: FrameSet) -> None:
        """Cheap: stores a reference. Drawing and encoding only happen for connected clients."""
        with self._lock:
            self._frames = frames

    def _latest(self) -> FrameSet | None:
        with self._lock:
            return self._frames

    def _annotated_combined(self) -> tuple[int, numpy.ndarray] | None:
        """The latest frame set as BGR with every pipeline's overlay, drawn once per frame."""
        frames = self._latest()
        if frames is None:
            return None
        with self._lock:
            if (
                self._annotated_cache is not None
                and self._annotated_cache[0] == frames.sequence
            ):
                return self._annotated_cache
        image = (
            cv2.cvtColor(frames.combined, cv2.COLOR_GRAY2BGR)
            if frames.combined.ndim == 2
            else frames.combined.copy()
        )
        for overlay in self._overlays():
            overlay(_split(image, frames.num_cameras))
        with self._lock:
            self._annotated_cache = (frames.sequence, image)
        return self._annotated_cache

    def _annotated_source(
        self, camera_index: int
    ) -> Callable[[], tuple[int, numpy.ndarray] | None]:
        def source() -> tuple[int, numpy.ndarray] | None:
            combined = self._annotated_combined()
            if combined is None:
                return None
            sequence, image = combined
            return sequence, _split(image, len(self.raw_cameras))[camera_index]

        return source

    def _raw_source(
        self, camera_index: int
    ) -> Callable[[], tuple[int, numpy.ndarray] | None]:
        def source() -> tuple[int, numpy.ndarray] | None:
            frames = self._latest()
            return (
                None
                if frames is None
                else (frames.sequence, frames.cameras[camera_index])
            )

        return source


def _split(image: numpy.ndarray, num_cameras: int) -> list[numpy.ndarray]:
    width = image.shape[1] // num_cameras
    return [image[:, i * width : (i + 1) * width] for i in range(num_cameras)]
