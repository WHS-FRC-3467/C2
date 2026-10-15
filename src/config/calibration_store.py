"""Per-camera calibrations that can be replaced while C2 runs.

Pipelines and the dashboard read calibrations from here every frame, so a
calibration saved by the calibration tool takes effect immediately, without
restarting C2.
"""

from __future__ import annotations

import logging
import shutil
import threading
import time
from collections.abc import Sequence
from pathlib import Path

import cv2

from config.local_config import CameraCalibration

log = logging.getLogger(__name__)


class CalibrationStore:
    def __init__(
        self, directory: Path, calibrations: Sequence[CameraCalibration | None]
    ) -> None:
        self._directory = directory
        self._lock = threading.Lock()
        self._calibrations = tuple(calibrations)

    def all(self) -> tuple[CameraCalibration | None, ...]:
        """Indexed by camera index. None where a camera has no calibration."""
        with self._lock:
            return self._calibrations

    def path(self, camera: int) -> Path:
        return self._directory / f"calibration_{camera}.yml"

    def save(
        self,
        camera: int,
        calibration: CameraCalibration,
        metadata: dict[str, str | float],
    ) -> Path:
        """Write calibration_<camera>.yml (keeping the old one as .previous.yml) and use it immediately."""
        path = self.path(camera)
        if path.exists():
            shutil.copy2(path, path.with_suffix(".previous.yml"))
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
        storage.write("calibration_time", time.strftime("%Y-%m-%d %H:%M:%S"))
        storage.write("image_width", calibration.image_width)
        storage.write("image_height", calibration.image_height)
        storage.write("camera_matrix", calibration.camera_matrix)
        storage.write("distortion_coefficients", calibration.distortion_coefficients)
        if calibration.reprojection_error is not None:
            storage.write("avg_reprojection_error", calibration.reprojection_error)
        for key, value in metadata.items():
            storage.write(key, value)
        storage.release()
        with self._lock:
            updated = list(self._calibrations)
            updated[camera] = calibration
            self._calibrations = tuple(updated)
        log.info("Saved and loaded new calibration for camera %d: %s", camera, path)
        return path
