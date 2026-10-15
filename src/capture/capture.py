"""What the capture loop needs from a frame source: a real camera or the simulator."""

from __future__ import annotations

from typing import Protocol

from capture.multi_camera_capture import CaptureSettings
from core.frames import FrameSet


class Capture(Protocol):
    def apply_settings(self, settings: CaptureSettings) -> None:
        """Apply /<dev>/config/* (device, resolution, exposure, gain)."""
        ...

    def read(self) -> FrameSet | None:
        """Block until the next frame set. None if not configured yet."""
        ...

    def snapshot(self) -> dict:
        """Current state, for the dashboard."""
        ...
