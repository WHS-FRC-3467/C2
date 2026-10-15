"""Camera data passed from the capture loop to pipelines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy


@dataclass(frozen=True)
class FrameSet:
    """One capture from the multi-camera module: every camera's image from the same instant.

    The arrays are read-only because every enabled pipeline shares them.
    Copy before drawing on them.
    """

    sequence: int
    """Increments by one for every capture. Gaps mean a pipeline skipped frames."""

    capture_time_us: int
    """When the frame was captured, in the coprocessor's NT time base (``ntcore._now()``).
    Pass it through ``TimeSync.stamp()`` to get the robot's time."""

    combined: numpy.ndarray
    """The raw side-by-side image from the module (height x width*num_cameras)."""

    cameras: tuple[numpy.ndarray, ...]
    """Per-camera views into ``combined``, indexed by camera index."""

    @property
    def num_cameras(self) -> int:
        return len(self.cameras)

    @staticmethod
    def from_combined(sequence: int, capture_time_us: int, combined: numpy.ndarray, num_cameras: int) -> FrameSet:
        combined.setflags(write=False)
        width = combined.shape[1] // num_cameras
        cameras = tuple(combined[:, i * width : (i + 1) * width] for i in range(num_cameras))
        return FrameSet(sequence, capture_time_us, combined, cameras)
