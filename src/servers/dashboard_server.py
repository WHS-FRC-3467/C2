"""The web dashboard on web_port: camera streams, pipelines, status, and logs.

Pipelines and capture are read-only here: the robot is the source of truth
for their configuration, and edits made here would be silently overwritten
when it republishes. The camera calibration tool is the exception: it is
local to the coprocessor.

    /                          the dashboard page (servers/dashboard/index.html)
    /api/status                everything on the page, as JSON
    /api/logs?after=<id>       recent log lines with id > after
    /stream                    every camera side by side, with overlays
    /cameras/<i>/stream        camera i with overlays
    /cameras/<i>/raw           camera i, no overlay

    /calibration               the calibration page (servers/dashboard/calibration.html)
    /calibration/stream        the camera being calibrated, with guidance drawn on it
    /api/calibration           calibration state, as JSON
    /api/calibration/board.png printable board (query: squares_x, squares_y, square_m, marker_m, dictionary)
    POST /api/calibration/{start,stop,capture,solve,continue,save}
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

import cv2
import ntcore
import numpy

from calibration.board import Board, BoardSpec
from calibration.mrcal_solver import MrcalError
from calibration.session import CalibrationSession
from capture.capture import Capture
from config.calibration_store import CalibrationStore
from config.local_config import LocalConfig
from core.log_buffer import LogBuffer
from core.status import StatusPublisher
from core.time_sync import TimeSync
from pipelines.runner import PipelineManager
from servers.debug_streams import DebugStreams
from servers.mjpeg_server import MjpegStream

log = logging.getLogger(__name__)

_PAGE = Path(__file__).with_name("dashboard") / "index.html"
_CALIBRATION_PAGE = Path(__file__).with_name("dashboard") / "calibration.html"
_CALIBRATION_ACTION = re.compile(
    r"^/api/calibration/(start|stop|capture|solve|continue|save)$"
)
_MAX_BODY_BYTES = 64 * 1024
_CAMERA_ROUTE = re.compile(r"^/cameras/(\d+)/(stream|raw)$")
_MAX_STRING = 300  # long settings (e.g. the tag layout JSON) are summarized


class DashboardServer:
    def __init__(
        self,
        config: LocalConfig,
        inst: ntcore.NetworkTableInstance,
        time_sync: TimeSync,
        status: StatusPublisher,
        capture: Capture,
        pipelines: PipelineManager,
        streams: DebugStreams,
        logs: LogBuffer,
        calibrations: CalibrationStore,
        calibration: CalibrationSession,
    ) -> None:
        self._config = config
        self._inst = inst
        self._time_sync = time_sync
        self._status = status
        self._capture = capture
        self._pipelines = pipelines
        self._streams = streams
        self._logs = logs
        self._calibrations = calibrations
        self.calibration = calibration
        self.calibration_stream = MjpegStream(calibration.latest_overlay, 85)

    def start(self) -> None:
        server = _DashboardHTTPServer(self._config.web_port, self)
        threading.Thread(
            target=server.serve_forever, name="dashboard", daemon=True
        ).start()
        log.info("Dashboard at http://0.0.0.0:%d", self._config.web_port)

    # ------------------------------------------------------------------ data

    def status(self) -> dict:
        health = self._status.snapshot()
        capture = self._capture.snapshot()
        settings = capture["settings"] or {}
        camera_size = (
            settings.get("camera_resolution_width"),
            settings.get("camera_resolution_height"),
        )
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
                for i, calibration in enumerate(self._calibrations.all())
            ],
            "pipelines": self._pipelines.snapshot(),
        }

    def calibration_action(self, action: str, body: dict) -> dict:
        """Run a calibration tool action. Raises ValueError (or MrcalError) with a user-facing message."""
        session = self.calibration
        if action == "start":
            board = (
                BoardSpec(**body.get("board", {}))
                if body.get("board")
                else session.board_spec
            )
            session.start(int(body.get("camera", 0)), board)
        elif action == "stop":
            session.stop()
        elif action == "capture":
            session.capture_now()
        elif action == "solve":
            session.solve_now()
        elif action == "continue":
            session.continue_collecting()
        elif action == "save":
            return {"ok": True, "path": session.save()}
        return {"ok": True}

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
        streams = (
            self._streams.annotated_cameras
            if kind == "stream"
            else self._streams.raw_cameras
        )
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
        elif url.path == "/calibration":
            self._send(200, "text/html; charset=utf-8", _CALIBRATION_PAGE.read_bytes())
        elif url.path == "/api/calibration":
            self._send_json(dashboard.calibration.snapshot())
        elif url.path == "/api/calibration/board.png":
            self._send_board_image(query, dashboard.calibration.board_spec)
        elif url.path == "/calibration/stream":
            dashboard.calibration_stream.serve(
                self, width=_int(query, "width"), max_fps=_float(query, "fps")
            )
        elif url.path == "/api/status":
            self._send_json(dashboard.status())
        elif url.path == "/api/logs":
            self._send_json(dashboard.logs_since(_int(query, "after") or 0))
        elif (stream := dashboard.stream_for(url.path)) is not None:
            stream.serve(self, width=_int(query, "width"), max_fps=_float(query, "fps"))
        else:
            self.send_error(404)

    def do_POST(self) -> None:
        match = _CALIBRATION_ACTION.match(urlparse(self.path).path)
        if match is None:
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > _MAX_BODY_BYTES:
            self.send_error(413)
            return
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
            self._send_json(self.dashboard.calibration_action(match.group(1), body))
        except (ValueError, TypeError, MrcalError) as e:
            self._send(
                400,
                "application/json",
                json.dumps({"ok": False, "error": str(e)}).encode(),
            )

    def _send_board_image(self, query: dict, default: BoardSpec) -> None:
        try:
            spec = BoardSpec(
                squares_x=_int(query, "squares_x") or default.squares_x,
                squares_y=_int(query, "squares_y") or default.squares_y,
                square_m=_float(query, "square_m") or default.square_m,
                marker_m=_float(query, "marker_m") or default.marker_m,
                dictionary=query.get("dictionary", [default.dictionary])[0],
            )
            ok, png = cv2.imencode(".png", Board(spec).image())
        except ValueError as e:
            self.send_error(400, str(e))
            return
        self._send(200, "image/png", png.tobytes())

    def _send_json(self, data: object) -> None:
        self._send(200, "application/json", json.dumps(_jsonable(data)).encode())

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(
        self, format: str, *args
    ) -> None:  # noqa: A002 - matches the base signature
        pass  # don't print a line per request


def _calibration_info(calibration, camera_size: tuple[int | None, int | None]) -> dict:
    if calibration is None:
        return {"loaded": False}
    size = (calibration.image_width, calibration.image_height)
    return {
        "loaded": True,
        "image_size": size if all(size) else None,
        "matches_resolution": (
            None if not all(size) or not all(camera_size) else size == camera_size
        ),
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
