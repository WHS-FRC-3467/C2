import threading
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer

import cv2

from config.config import ConfigStore

JPEG_QUALITY = 80
RAW_JPEG_QUALITY = 95


class StreamServer:
    """Interface for outputing camera frames."""

    def start(self, config_store: ConfigStore) -> None:
        raise NotImplementedError

    def set_frame(self, frame: cv2.Mat) -> None:
        raise NotImplementedError


class MjpegServer(StreamServer):
    _frame = None
    _has_frame: bool = False
    _lock: threading.Lock = threading.Lock()

    def _make_handler(self_mjpeg):  # type: ignore
        class MJPEGHandler(BaseHTTPRequestHandler):
            HTML = b'<html><body style="margin:0;background:#000"><img src="/stream" style="width:100%;height:auto"></body></html>'

            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(self.HTML)))
                    self.end_headers()
                    self.wfile.write(self.HTML)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
                    try:
                        while True:
                            with self_mjpeg._lock:
                                frame = self_mjpeg._frame
                            if frame is None:
                                import time
                                time.sleep(0.01)
                                continue
                            ret, jpeg = cv2.imencode('.jpg', frame, encode_params)
                            if not ret:
                                continue
                            data = jpeg.tobytes()
                            self.wfile.write(b'--frame\r\n')
                            self.wfile.write(b'Content-Type: image/jpeg\r\n')
                            self.wfile.write(f'Content-Length: {len(data)}\r\n'.encode())
                            self.wfile.write(b'\r\n')
                            self.wfile.write(data)
                            self.wfile.write(b'\r\n')
                    except BrokenPipeError:
                        return
                else:
                    self.send_error(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass

        return MJPEGHandler

    def _run(self, port: int) -> None:
        server = HTTPServer(("", port), self._make_handler())
        server.serve_forever()

    def start(self, config_store: ConfigStore) -> None:
        threading.Thread(target=self._run, daemon=True, args=(config_store.local_config.stream_port,)).start()

    def set_frame(self, frame: cv2.Mat) -> None:
        with self._lock:
            self._frame = frame.copy()


class RawCameraMjpegServer:
    """Per-camera MJPEG server that serves raw frames without overlays.

    Only encodes JPEG when at least one client is connected.
    """

    def __init__(self, camera_index: int):
        self.camera_index = camera_index
        self._frame = None
        self._lock = threading.Lock()
        self._client_count = 0
        self._client_count_lock = threading.Lock()

    @property
    def has_clients(self) -> bool:
        with self._client_count_lock:
            return self._client_count > 0

    def _make_handler(self_raw):  # type: ignore
        class RawMJPEGHandler(BaseHTTPRequestHandler):
            HTML = b'<html><body style="margin:0;background:#000"><img src="/stream" style="width:100%;height:auto"></body></html>'

            def do_GET(self):
                if self.path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(self.HTML)))
                    self.end_headers()
                    self.wfile.write(self.HTML)
                elif self.path == "/stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    encode_params = [cv2.IMWRITE_JPEG_QUALITY, RAW_JPEG_QUALITY]
                    with self_raw._client_count_lock:
                        self_raw._client_count += 1
                    try:
                        while True:
                            with self_raw._lock:
                                frame = self_raw._frame
                            if frame is None:
                                import time
                                time.sleep(0.01)
                                continue
                            ret, jpeg = cv2.imencode('.jpg', frame, encode_params)
                            if not ret:
                                continue
                            data = jpeg.tobytes()
                            self.wfile.write(b'--frame\r\n')
                            self.wfile.write(b'Content-Type: image/jpeg\r\n')
                            self.wfile.write(f'Content-Length: {len(data)}\r\n'.encode())
                            self.wfile.write(b'\r\n')
                            self.wfile.write(data)
                            self.wfile.write(b'\r\n')
                    except BrokenPipeError:
                        pass
                    finally:
                        with self_raw._client_count_lock:
                            self_raw._client_count -= 1
                else:
                    self.send_error(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass

        return RawMJPEGHandler

    def start(self, port: int) -> None:
        def run():
            server = ThreadingHTTPServer(("", port), self._make_handler())
            server.serve_forever()
        threading.Thread(target=run, daemon=True).start()

    def set_frame(self, frame: cv2.Mat) -> None:
        if not self.has_clients:
            return
        with self._lock:
            self._frame = frame.copy()
