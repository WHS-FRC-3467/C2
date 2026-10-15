"""MJPEG-over-HTTP streams. Frames are only drawn and encoded while someone is watching."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler

import cv2
import numpy

FrameSource = Callable[[], "tuple[int, numpy.ndarray] | None"]
"""Returns (sequence number, image), or None if there is nothing to show yet.
Called only while a client is connected, so it can do the drawing lazily."""

_POLL_S = 0.005


class MjpegStream:
    """One image source, servable to any number of HTTP clients at their own size and rate."""

    def __init__(self, source: FrameSource, jpeg_quality: int = 80) -> None:
        self._source = source
        self._quality = jpeg_quality
        # Encode each (frame, size) once, however many clients are watching.
        self._lock = threading.Lock()
        self._cache: dict[int | None, tuple[int, bytes]] = {}

    def serve(
        self,
        handler: BaseHTTPRequestHandler,
        width: int | None = None,
        max_fps: float | None = None,
    ) -> None:
        """Stream to ``handler`` until the client disconnects.

        ``width`` downscales (keeping the aspect ratio) to save bandwidth;
        ``max_fps`` caps the frame rate.
        """
        handler.send_response(200)
        handler.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        handler.send_header("Cache-Control", "no-store")
        handler.end_headers()
        # Each part is followed immediately by the next boundary: browsers only paint a
        # part once they see the boundary after it, so otherwise the last frame of a
        # stream that stops (e.g. after a calibration solve) would never appear.
        min_interval_s = 1.0 / max_fps if max_fps else 0.0
        last_sequence = -1
        last_sent_s = 0.0
        try:
            handler.wfile.write(b"--frame\r\n")
            while True:
                wait_s = last_sent_s + min_interval_s - time.monotonic()
                if wait_s > 0:
                    time.sleep(wait_s)
                frame = self._jpeg_after(last_sequence, width)
                if frame is None:
                    time.sleep(_POLL_S)
                    continue
                last_sequence, data = frame
                last_sent_s = time.monotonic()
                handler.wfile.write(
                    b"Content-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n"
                    % len(data)
                )
                handler.wfile.write(data)
                handler.wfile.write(b"\r\n--frame\r\n")
                handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _jpeg_after(
        self, last_sequence: int, width: int | None
    ) -> tuple[int, bytes] | None:
        """The newest frame as JPEG at ``width``, if it is newer than ``last_sequence``."""
        with self._lock:
            cached = self._cache.get(width)
            if cached is not None and cached[0] > last_sequence:
                return cached
            frame = self._source()
            if frame is None or frame[0] <= last_sequence:
                return None
            sequence, image = frame
            if width is not None and 0 < width < image.shape[1]:
                height = max(1, round(image.shape[0] * width / image.shape[1]))
                image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, self._quality]
            )
            if not ok:
                return None
            self._cache[width] = (sequence, jpeg.tobytes())
            return self._cache[width]
