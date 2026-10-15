"""The web dashboard on web_port: camera streams, pipelines, status, and logs.

Read-only by design: the robot is the source of truth for configuration, and
edits made here would be silently overwritten when it republishes.

    /                          the dashboard page (servers/dashboard/index.html)
    /api/status                everything on the page, as JSON
    /api/logs?after=<id>       recent log lines with id > after
    /stream                    every camera side by side, with overlays
    /cameras/<i>/stream        camera i with overlays
    /cameras/<i>/raw           camera i, no overlay
Streams take ?width=<px> (downscale) and ?fps=<max>, to save bandwidth over
the robot radio.
"""

from __future__ import annotations

import json
import logging
import math
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import ntcore
import numpy

from capture.multi_camera_capture import MultiCameraCapture
from config.local_config import LocalConfig
from core.log_buffer import LogBuffer
from core.status import StatusPublisher
from core.time_sync import TimeSync
from pipelines.runner import PipelineManager
from servers.debug_streams import DebugStreams
from servers.mjpeg_server import MjpegStream

log = logging.getLogger(__name__)

_PAGE = Path(__file__).with_name("dashboard") / "index.html"
_CAMERA_ROUTE = re.compile(r"^/cameras/(\d+)/(stream|raw)$")
_MAX_STRING = 300  # long settings (e.g. the tag layout JSON) are summarized


class DashboardServer:
    def __init__(
        self,
        config: LocalConfig,
        inst: ntcore.NetworkTableInstance,
        time_sync: TimeSync,
        status: StatusPublisher,
        capture: MultiCameraCapture,
        pipelines: PipelineManager,
        streams: DebugStreams,
        logs: LogBuffer,
    ) -> None:
        self._config = config
        self._inst = inst
        self._time_sync = time_sync
        self._status = status
        self._capture = capture
        self._pipelines = pipelines
        self._streams = streams
        self._logs = logs

    def start(self) -> None:
        server = _DashboardHTTPServer(self._config.web_port, self)
        threading.Thread(target=server.serve_forever, name="dashboard", daemon=True).start()
        log.info("Dashboard at http://0.0.0.0:%d", self._config.web_port)

    # ------------------------------------------------------------------ data

    def status(self) -> dict:
        health = self._status.snapshot()
        capture = self._capture.snapshot()
        settings = capture["settings"] or {}
        camera_size = (settings.get("camera_resolution_width"), settings.get("camera_resolution_height"))
        return {
            "device": {
                "id": self._config.device_id,
                "server_ip": self._config.server_ip,
                "num_cameras": self._config.num_cameras,
            },
            "nt": {
                "connected": self._inst.isConnected(),
                "connections": [
                    {
                        "remote_id": c.remote_id,
                        "remote_ip": c.remote_ip,
                        "remote_port": c.remote_port,
                        "protocol": c.protocol_version,
                    }
                    for c in self._inst.getConnections()
                ],
            },
            "time_sync": self._time_sync.snapshot(),
            "health": health,
            "capture": capture,
            "cameras": [
                {
                    "index": i,
                    "connected": health["cameras_connected"][i],
                    "calibration": _calibration_info(calibration, camera_size),
                }
                for i, calibration in enumerate(self._config.calibrations)
            ],
            "pipelines": self._pipelines.snapshot(),
        }

    def logs_since(self, after_id: int) -> list[dict]:
        return self._logs.since(after_id)

    def stream_for(self, path: str) -> MjpegStream | None:
        """The stream served at ``path``, or None if there isn't one."""
        if path == "/stream":
            return self._streams.annotated
        match = _CAMERA_ROUTE.match(path)
        if match is None:
            return None
        index, kind = int(match.group(1)), match.group(2)
        streams = self._streams.annotated_cameras if kind == "stream" else self._streams.raw_cameras
        return streams[index] if index < len(streams) else None


class _DashboardHTTPServer(ThreadingHTTPServer):
    """Carries the dashboard so request handlers can reach it as ``self.server.dashboard``."""

    daemon_threads = True

    def __init__(self, port: int, dashboard: DashboardServer) -> None:
        super().__init__(("", port), _RequestHandler)
        self.dashboard = dashboard


class _RequestHandler(BaseHTTPRequestHandler):
    """One instance per request; ``http.server`` creates it and sets ``self.server``."""

    @property
    def dashboard(self) -> DashboardServer:
        assert isinstance(self.server, _DashboardHTTPServer)
        return self.server.dashboard

    def do_GET(self) -> None:
        dashboard = self.dashboard
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            self._send(200, "text/html; charset=utf-8", _PAGE.read_bytes())
        elif url.path == "/api/status":
            self._send_json(dashboard.status())
        elif url.path == "/api/logs":
            self._send_json(dashboard.logs_since(_int(query, "after") or 0))
        elif (stream := dashboard.stream_for(url.path)) is not None:
            stream.serve(self, width=_int(query, "width"), max_fps=_float(query, "fps"))
        else:
            self.send_error(404)

    def _send_json(self, data: object) -> None:
        self._send(200, "application/json", json.dumps(_jsonable(data)).encode())

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - matches the base signature
        pass  # don't print a line per request


def _calibration_info(calibration, camera_size: tuple[int | None, int | None]) -> dict:
    if calibration is None:
        return {"loaded": False}
    size = (calibration.image_width, calibration.image_height)
    return {
        "loaded": True,
        "image_size": size if all(size) else None,
        "matches_resolution": None if not all(size) or not all(camera_size) else size == camera_size,
        "reprojection_error": calibration.reprojection_error,
        "fx": float(calibration.camera_matrix[0, 0]),
        "fy": float(calibration.camera_matrix[1, 1]),
        "cx": float(calibration.camera_matrix[0, 2]),
        "cy": float(calibration.camera_matrix[1, 2]),
    }


def _jsonable(value: object) -> object:
    """Make data safe for JSON.parse: no NaN/inf, no numpy types, long strings summarized."""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, numpy.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, str) and len(value) > _MAX_STRING:
        return f"{value[:80]}… ({len(value)} chars)"
    return value


def _int(query: dict, key: str) -> int | None:
    try:
        return int(query[key][0])
    except (KeyError, ValueError):
        return None


def _float(query: dict, key: str) -> float | None:
    try:
        value = float(query[key][0])
        return value if value > 0 else None
    except (KeyError, ValueError):
        return None
