"""MJPEG-over-HTTP debug streams. Frames are only encoded while someone is watching."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy

log = logging.getLogger(__name__)

FrameSource = Callable[[], "tuple[int, numpy.ndarray] | None"]
"""Returns (sequence number, image), or None if there is nothing to show yet.
Called only while a client is connected, so it can do the drawing lazily."""

_INDEX_HTML = b'<html><body style="margin:0;background:#000"><img src="/stream" style="width:100%;height:auto"></body></html>'
_POLL_S = 0.005


class MjpegServer:
    def __init__(self, port: int, source: FrameSource, jpeg_quality: int = 80) -> None:
        self.port = port
        self._source = source
        self._quality = jpeg_quality
        # Encode each frame once, however many clients are watching.
        self._lock = threading.Lock()
        self._cached: tuple[int, bytes] | None = None

    def start(self) -> None:
        server = ThreadingHTTPServer(("", self.port), self._handler_class())
        server.daemon_threads = True
        threading.Thread(target=server.serve_forever, name=f"mjpeg-{self.port}", daemon=True).start()
        log.info("MJPEG stream at http://0.0.0.0:%d", self.port)

    def _jpeg_after(self, last_sequence: int) -> tuple[int, bytes] | None:
        """The newest frame as JPEG, if it is newer than ``last_sequence``."""
        with self._lock:
            if self._cached is not None and self._cached[0] > last_sequence:
                return self._cached
            frame = self._source()
            if frame is None or frame[0] <= last_sequence:
                return None
            ok, jpeg = cv2.imencode(".jpg", frame[1], [cv2.IMWRITE_JPEG_QUALITY, self._quality])
            if not ok:
                return None
            self._cached = (frame[0], jpeg.tobytes())
            return self._cached

    def _handler_class(server_self) -> type[BaseHTTPRequestHandler]:  # noqa: N805
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(_INDEX_HTML)))
                    self.end_headers()
                    self.wfile.write(_INDEX_HTML)
                elif self.path == "/stream":
                    self._stream()
                else:
                    self.send_error(404)

            def _stream(self) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                last_sequence = -1
                try:
                    while True:
                        frame = server_self._jpeg_after(last_sequence)
                        if frame is None:
                            time.sleep(_POLL_S)
                            continue
                        last_sequence, data = frame
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(data)
                        )
                        self.wfile.write(data)
                        self.wfile.write(b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, format: str, *args) -> None:  # noqa: A002
                pass

        return Handler
