"""The robot's true pose over time, as published by the robot simulation."""

from __future__ import annotations

import bisect

from wpimath.geometry import Pose3d, Twist3d


class PoseHistory:
    """Timestamped poses, interpolated to any instant (times in C2's NT time base, us)."""

    def __init__(self, max_age_us: int = 2_000_000) -> None:
        self._max_age_us = max_age_us
        self._times: list[int] = []
        self._poses: list[Pose3d] = []

    def add(self, time_us: int, pose: Pose3d) -> None:
        index = bisect.bisect(self._times, time_us)
        self._times.insert(index, time_us)
        self._poses.insert(index, pose)
        cutoff = self._times[-1] - self._max_age_us
        while self._times and self._times[0] < cutoff:
            self._times.pop(0)
            self._poses.pop(0)

    @property
    def latest_time_us(self) -> int | None:
        return self._times[-1] if self._times else None

    def sample(self, time_us: int) -> Pose3d | None:
        """The pose at ``time_us``, interpolated between samples, or held at the ends."""
        if not self._times:
            return None
        index = bisect.bisect(self._times, time_us)
        if index == 0:
            return self._poses[0]
        if index == len(self._times):
            return self._poses[-1]
        t0, t1 = self._times[index - 1], self._times[index]
        p0, p1 = self._poses[index - 1], self._poses[index]
        fraction = (time_us - t0) / (t1 - t0) if t1 != t0 else 0.0
        # Same as WPILib's Pose3d.interpolate: move along the twist between them.
        twist = p0.log(p1)
        return p0.exp(
            Twist3d(
                *(
                    fraction * v
                    for v in (
                        twist.dx,
                        twist.dy,
                        twist.dz,
                        twist.rx,
                        twist.ry,
                        twist.rz,
                    )
                )
            )
        )
