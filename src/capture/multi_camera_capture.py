"""Capture from a multi-camera V4L2 module that outputs one wide, side-by-side frame."""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy

from core.frames import FrameSet
from core.time_sync import monotonic_to_nt_us

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureSettings:
    """Device-wide capture settings, at /<device_id>/config/<field> in NT."""

    camera_id: str = ""
    camera_resolution_width: int = 0
    """Width of ONE camera. The module's frame is num_cameras times wider."""
    camera_resolution_height: int = 0
    camera_exposure: int = 0
    camera_gain: int = 0


class MultiCameraCapture:
    """Reads grayscale frames from the module and timestamps them.

    The timestamp comes from the V4L2 buffer (CLOCK_MONOTONIC, set by the
    driver when the frame arrived), not from when ``read()`` returned. That
    excludes the time spent waiting in the kernel queue and copying the frame
    out, which is ~30 ms on a typical USB camera.
    """

    FPS = 45

    # Exit (and let systemd restart us) if the camera freezes, e.g. unplugged.
    WATCHDOG_SLOW_FRAME_S = 2.0
    WATCHDOG_MAX_SLOW_FRAMES = 3
    WATCHDOG_WINDOW_S = 15.0

    # A V4L2 timestamp older than this, or in the future, is not trusted.
    MAX_BUFFER_AGE_S = 1.0

    def __init__(self, num_cameras: int) -> None:
        self._num_cameras = num_cameras
        self._video: cv2.VideoCapture | None = None
        self._settings: CaptureSettings | None = None
        self._sequence = 0
        self._slow_frames: deque[float] = deque()
        self._warned_bad_timestamp = False

    def apply_settings(self, settings: CaptureSettings) -> None:
        """Apply new settings. Reopens the camera if the device or resolution changed."""
        old = self._settings
        self._settings = settings
        if old is None or (old.camera_id, old.camera_resolution_width, old.camera_resolution_height) != (
            settings.camera_id,
            settings.camera_resolution_width,
            settings.camera_resolution_height,
        ):
            self._close()
            return  # exposure/gain get applied after the next open
        if self._video is not None and (old.camera_exposure, old.camera_gain) != (
            settings.camera_exposure,
            settings.camera_gain,
        ):
            self._apply_exposure()

    def read(self) -> FrameSet | None:
        """Block until the next frame. Returns None if the camera isn't configured yet."""
        if self._video is None and not self._open():
            return None
        assert self._video is not None

        read_start = time.monotonic()
        ok, image = self._video.read()
        read_end = time.monotonic()
        if not ok:
            log.error("Capture failed, exiting for restart")
            self._close()
            sys.exit(1)
        self._watchdog(read_end - read_start)
        if image.ndim == 3 and image.shape[2] == 2:
            # Camera ignored the GREY request and sent raw YUYV (e.g. a USB
            # webcam on a dev machine). The Y channel is the grayscale image.
            image = numpy.ascontiguousarray(image[:, :, 0])

        self._sequence += 1
        return FrameSet.from_combined(self._sequence, self._capture_time_us(read_end), image, self._num_cameras)

    # --------------------------------------------------------------- private

    def _capture_time_us(self, read_end_monotonic_s: float) -> int:
        """When the frame was captured, in the NT time base."""
        buffer_ms = self._video.get(cv2.CAP_PROP_POS_MSEC) if self._video is not None else 0.0
        if buffer_ms > 0:
            age_s = time.monotonic() - buffer_ms / 1000.0
            if 0.0 <= age_s <= self.MAX_BUFFER_AGE_S:
                return monotonic_to_nt_us(buffer_ms / 1000.0)
        if not self._warned_bad_timestamp:
            self._warned_bad_timestamp = True
            log.warning(
                "V4L2 buffer timestamp unusable (%.1f ms); falling back to read() return time. "
                "Timestamps will run late by the driver's queueing delay.",
                buffer_ms,
            )
        return monotonic_to_nt_us(read_end_monotonic_s)

    def _open(self) -> bool:
        settings = self._settings
        if settings is None or settings.camera_id == "":
            return False
        total_width = settings.camera_resolution_width * self._num_cameras
        height = settings.camera_resolution_height
        log.info(
            "Opening camera %s (%d cameras, %dx%d)", settings.camera_id, self._num_cameras, total_width, height
        )
        device: int | str = int(settings.camera_id) if settings.camera_id.isdigit() else settings.camera_id
        video = cv2.VideoCapture(device, cv2.CAP_V4L2)
        video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"GREY"))
        video.set(cv2.CAP_PROP_CONVERT_RGB, 0)
        video.set(cv2.CAP_PROP_FRAME_WIDTH, total_width)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        video.set(cv2.CAP_PROP_FPS, self.FPS)
        if not video.isOpened():
            log.error("Could not open camera %s", settings.camera_id)
            return False
        self._video = video
        # v4l2-ctl only sticks once the stream is running, so grab a frame first.
        video.grab()
        self._apply_exposure()
        return True

    def _close(self) -> None:
        if self._video is not None:
            self._video.release()
            self._video = None

    def _apply_exposure(self) -> None:
        assert self._settings is not None
        device = self._settings.camera_id  # v4l2-ctl reads "0" as /dev/video0
        _v4l2_set(device, "exposure", self._settings.camera_exposure)
        _v4l2_set(device, "analogue_gain", self._settings.camera_gain)

    def _watchdog(self, read_duration_s: float) -> None:
        if read_duration_s <= self.WATCHDOG_SLOW_FRAME_S:
            return
        now = time.monotonic()
        self._slow_frames.append(now)
        while self._slow_frames and self._slow_frames[0] < now - self.WATCHDOG_WINDOW_S:
            self._slow_frames.popleft()
        log.warning(
            "Slow frame (%.2f s), %d/%d in the last %.0f s",
            read_duration_s,
            len(self._slow_frames),
            self.WATCHDOG_MAX_SLOW_FRAMES,
            self.WATCHDOG_WINDOW_S,
        )
        if len(self._slow_frames) >= self.WATCHDOG_MAX_SLOW_FRAMES:
            log.error("Camera appears frozen, exiting for restart")
            self._close()
            sys.exit(1)


def _v4l2_set(device: str, control: str, value: int) -> None:
    try:
        result = subprocess.run(
            ["v4l2-ctl", "-d", device, "--set-ctrl", f"{control}={value}"], capture_output=True, text=True
        )
    except FileNotFoundError:
        log.error("v4l2-ctl not installed; can't set %s (apt install v4l-utils)", control)
        return
    if result.returncode != 0:
        log.error("v4l2-ctl %s=%d failed: %s", control, value, result.stderr.strip())
    else:
        log.info("v4l2-ctl %s=%d", control, value)
