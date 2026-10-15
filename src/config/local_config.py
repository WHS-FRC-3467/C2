"""Settings stored on the coprocessor itself (config.json and calibration_<i>.yml).

These are the things needed before NetworkTables is up (who to connect to,
which ports to serve on), plus per-camera intrinsics, which belong to the
hardware rather than to the robot program.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: numpy.ndarray
    distortion_coefficients: numpy.ndarray
    image_width: int = 0
    """Resolution the calibration was made at (0 if the file doesn't say).
    Intrinsics are only valid at that resolution."""
    image_height: int = 0
    reprojection_error: float | None = None


@dataclass(frozen=True)
class LocalConfig:
    device_id: str
    """NT root table for this coprocessor, e.g. "dsv0" -> /dsv0/..."""

    server_ip: str

    stream_port: int = 8000
    """Annotated stream port. Raw camera i is served on stream_port + 1 + i,
    and the calibration upload page on stream_port + 10."""

    num_cameras: int = 4

    # Capture defaults, used until the robot publishes /<device_id>/config/*.
    camera_id: str = ""
    camera_resolution_width: int = 0
    camera_resolution_height: int = 0
    camera_exposure: int = 0
    camera_gain: int = 0

    time_sync_probe_interval_s: float = 10.0
    """How often to re-measure the robot clock offset (see core/time_sync.py). 0 disables."""

    pipelines_enabled: dict[str, bool] = field(default_factory=dict)
    """Per-pipeline enabled default, used until NT says otherwise, e.g. {"apriltag": true}."""

    calibrations: tuple[CameraCalibration | None, ...] = ()
    """Indexed by camera index. None if calibration_<i>.yml is missing or invalid."""


def load_local_config(config_path: Path, calibration_dir: Path) -> LocalConfig:
    data = json.loads(config_path.read_text())
    num_cameras = int(data.get("num_cameras", 4))
    return LocalConfig(
        device_id=data["device_id"],
        server_ip=data["server_ip"],
        stream_port=int(data.get("stream_port", 8000)),
        num_cameras=num_cameras,
        camera_id=str(data.get("camera_id", "")),
        camera_resolution_width=int(data.get("camera_resolution_width", 0)),
        camera_resolution_height=int(data.get("camera_resolution_height", 0)),
        camera_exposure=int(data.get("camera_exposure", 0)),
        camera_gain=int(data.get("camera_gain", 0)),
        time_sync_probe_interval_s=float(data.get("time_sync_probe_interval_s", 10.0)),
        pipelines_enabled={name: bool(v) for name, v in data.get("pipelines_enabled", {}).items()},
        calibrations=tuple(
            _load_calibration(calibration_dir / f"calibration_{i}.yml") for i in range(num_cameras)
        ),
    )


def _load_calibration(path: Path) -> CameraCalibration | None:
    if not path.exists():
        log.warning("No calibration at %s", path)
        return None
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    camera_matrix = storage.getNode("camera_matrix").mat()
    distortion = storage.getNode("distortion_coefficients").mat()
    width = int(storage.getNode("image_width").real())
    height = int(storage.getNode("image_height").real())
    error_node = storage.getNode("avg_reprojection_error")
    error = None if error_node.empty() else error_node.real()
    storage.release()
    if not isinstance(camera_matrix, numpy.ndarray) or not isinstance(distortion, numpy.ndarray):
        log.warning("Calibration %s is missing camera_matrix or distortion_coefficients", path)
        return None
    return CameraCalibration(camera_matrix, distortion, width, height, error)
