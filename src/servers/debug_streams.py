"""The HTTP debug streams: one annotated view of every camera, plus one raw stream per camera.

Ports: stream_port shows every camera side by side with each enabled
pipeline's overlay drawn on top. stream_port + 1 + i shows camera i raw,
with no overlay (handy for focusing and checking exposure).
"""

from __future__ import annotations

import threading
from collections.abc import Callable

import cv2
import numpy

from core.frames import FrameSet
from pipelines.pipeline import Overlay
from servers.mjpeg_server import MjpegServer

ANNOTATED_JPEG_QUALITY = 80
RAW_JPEG_QUALITY = 95


class DebugStreams:
    def __init__(self, base_port: int, num_cameras: int, overlays: Callable[[], list[Overlay]]) -> None:
        self._overlays = overlays
        self._lock = threading.Lock()
        self._frames: FrameSet | None = None
        self._servers = [MjpegServer(base_port, self._annotated, ANNOTATED_JPEG_QUALITY)] + [
            MjpegServer(base_port + 1 + i, self._raw_source(i), RAW_JPEG_QUALITY) for i in range(num_cameras)
        ]

    def start(self) -> None:
        for server in self._servers:
            server.start()

    def set_frames(self, frames: FrameSet) -> None:
        """Cheap: stores a reference. Drawing and encoding only happen for connected clients."""
        with self._lock:
            self._frames = frames

    def _latest(self) -> FrameSet | None:
        with self._lock:
            return self._frames

    def _annotated(self) -> tuple[int, numpy.ndarray] | None:
        frames = self._latest()
        if frames is None:
            return None
        image = cv2.cvtColor(frames.combined, cv2.COLOR_GRAY2BGR) if frames.combined.ndim == 2 else frames.combined.copy()
        width = image.shape[1] // frames.num_cameras
        cameras = [image[:, i * width : (i + 1) * width] for i in range(frames.num_cameras)]
        for overlay in self._overlays():
            overlay(cameras)
        return frames.sequence, image

    def _raw_source(self, camera_index: int) -> Callable[[], tuple[int, numpy.ndarray] | None]:
        def source() -> tuple[int, numpy.ndarray] | None:
            frames = self._latest()
            return None if frames is None else (frames.sequence, frames.cameras[camera_index])

        return source
