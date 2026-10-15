"""AprilTag pipeline: per-camera field-relative pose from AprilTags.

Inputs:  every camera of the multi-camera module.
Outputs: /<dev>/pipelines/apriltag/output/camera_<i>   (schema/apriltag.fbs, "dsv0_fb")
         One message per camera per frame, including frames with no tags,
         so the robot can tell "no tags seen" apart from "disconnected".
Config:  /<dev>/pipelines/apriltag/config/tag_layout       AprilTagFieldLayout JSON
         /<dev>/pipelines/apriltag/config/fiducial_size_m  tag edge length, meters
         /<dev>/pipelines/apriltag/config/min_tag_size_px     leave small (distant) tags out of the solve
         /<dev>/pipelines/apriltag/config/temporal_acceleration, roi_padding, full_frame_interval
             only search where tags should be (see temporal.py)
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy

from pipelines.apriltag.detector import ArucoNanoDetector, TagDetection
from pipelines.apriltag.encoder import TYPE_STRING, encode_camera_frame
from pipelines.apriltag.pose_solver import (
    CameraPoseObservation,
    PoseSolution,
    solve_camera_pose,
)
from pipelines.apriltag.tag_filter import exclude_small_tags
from pipelines.apriltag.tag_layout import FieldTag, parse_tag_layout
from pipelines.apriltag.temporal import (
    CameraTracker,
    SearchReport,
    TagGeometry,
    TrackingSettings,
)
from pipelines.pipeline import Overlay, Pipeline, PipelineContext, PipelineInputs
from vision.overlay import draw_tag_detections

log = logging.getLogger(__name__)

_MAX_TRUSTED_AMBIGUITY = 0.2
"""Single-tag poses with best/alternate error ratio above this aren't used for tracking."""


@dataclass(frozen=True)
class AprilTagSettings:
    tag_layout: str = ""
    fiducial_size_m: float = 0.1651  # 6.5 in
    min_tag_size_px: float = 0.0
    """Tags whose longest edge is shorter than this in the image are left out
    of the solve. 0 = use every tag. See tag_filter.py."""
    temporal_acceleration: bool = True
    """Search only where tags are predicted to be, from recent poses. See temporal.py."""
    roi_padding: float = 0.5
    """Padding around each predicted tag, as a fraction of its size in the image."""
    full_frame_interval: int = 15
    """Search the whole image at least every N frames (finds tags the prediction missed)."""


class AprilTagPipeline(Pipeline[AprilTagSettings]):
    NAME = "apriltag"
    SETTINGS = AprilTagSettings

    def __init__(self, context: PipelineContext) -> None:
        super().__init__(context)
        n = context.num_cameras
        self._outputs = [context.output(f"camera_{i}", TYPE_STRING) for i in range(n)]
        # aruco_nano releases the GIL, so one thread per camera runs in parallel.
        self._pool = ThreadPoolExecutor(max_workers=n, thread_name_prefix="apriltag")
        self._detectors = [ArucoNanoDetector() for _ in range(n)]
        self._trackers = [CameraTracker() for _ in range(n)]
        self._tracking = TrackingSettings(
            enabled=False, padding=0.5, full_frame_interval=0
        )
        self._layout: dict[int, FieldTag] = {}
        self._tag_geometry = TagGeometry({})
        self._tag_size_m = 0.0
        self._min_tag_size_px = 0.0
        self._fps = _FpsCounter()
        self._summary: dict | None = None

    def configure(self, settings: AprilTagSettings) -> None:
        self._tag_size_m = settings.fiducial_size_m
        self._min_tag_size_px = settings.min_tag_size_px
        self._tracking = TrackingSettings(
            enabled=settings.temporal_acceleration,
            padding=max(0.0, settings.roi_padding),
            full_frame_interval=max(0, settings.full_frame_interval),
        )
        self._layout = _load_layout(settings)
        self._tag_geometry = TagGeometry(self._layout)

    def process(self, inputs: PipelineInputs) -> Overlay | None:
        frames = inputs.frames
        assert frames is not None
        fps = self._fps.tick()

        calibrations = self.context.calibrations
        detections = list(
            self._pool.map(
                lambda i: self._trackers[i].detect(
                    self._detectors[i],
                    frames.cameras[i],
                    frames.capture_time_us,
                    self._tag_geometry,
                    calibrations[i],
                    self._tracking,
                ),
                range(frames.num_cameras),
            )
        )

        # None if we aren't synced with the robot: skip solving, nothing can be sent.
        stamp = self.context.time_sync.stamp(frames.capture_time_us)
        observations: list[CameraPoseObservation | None] = [None] * frames.num_cameras
        excluded: list[list[TagDetection]] = [[] for _ in range(frames.num_cameras)]
        if stamp is not None:
            for i, camera_detections in enumerate(detections):
                calibration = self.context.calibrations[i]
                if calibration is not None and self._layout:
                    used, excluded[i] = exclude_small_tags(
                        camera_detections, self._min_tag_size_px
                    )
                    observations[i] = solve_camera_pose(
                        used, self._layout, self._tag_size_m, calibration
                    )
                self._outputs[i].publish(
                    encode_camera_frame(i, observations[i], fps), stamp
                )
        for tracker, observation in zip(self._trackers, observations):
            tracker.record(frames.capture_time_us, _trusted_pose(observation))

        self._summary = {
            "published": stamp is not None,
            "layout_tags": len(self._layout),
            "tag_size_m": self._tag_size_m,
            "cameras": [
                _camera_summary(
                    i,
                    detections[i],
                    excluded[i],
                    observations[i],
                    calibrations[i] is not None,
                    self._trackers[i].last_report,
                )
                for i in range(frames.num_cameras)
            ],
        }
        return _overlay(detections, [t.last_report for t in self._trackers])

    def summary(self) -> dict | None:
        return self._summary

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        for detector in self._detectors:
            detector.close()


def _trusted_pose(observation: CameraPoseObservation | None):
    """The camera pose, if it's reliable enough to predict where tags will be next frame.

    A lone tag has two candidate poses; if they fit almost equally well the best
    one may be the wrong one, and predicting from it (and extrapolating its
    jump as motion) would search the wrong places. Measured in simulation at
    2 m/s + 4 rad/s: recall 93.7% using every pose, 97.2% with this check.
    """
    if observation is None:
        return None
    if observation.alternate is not None and (
        observation.best.reprojection_error
        >= _MAX_TRUSTED_AMBIGUITY * observation.alternate.reprojection_error
    ):
        return None
    return observation.best.field_to_camera


def _load_layout(settings: AprilTagSettings) -> dict[int, FieldTag]:
    if not settings.tag_layout:
        log.warning("No tag layout yet; publishing empty results")
        return {}
    try:
        layout = parse_tag_layout(settings.tag_layout, settings.fiducial_size_m)
    except ValueError:
        log.exception("Ignoring tag layout; publishing empty results")
        return {}
    log.info("Loaded tag layout with %d tags", len(layout))
    return layout


def _camera_summary(
    index: int,
    detections: list[TagDetection],
    excluded: list[TagDetection],
    observation: CameraPoseObservation | None,
    calibrated: bool,
    search: SearchReport,
) -> dict:
    def pose(solution: PoseSolution) -> dict:
        p = solution.field_to_camera
        r = p.rotation()
        return {
            "x": p.X(),
            "y": p.Y(),
            "z": p.Z(),
            "roll_deg": math.degrees(r.X()),
            "pitch_deg": math.degrees(r.Y()),
            "yaw_deg": math.degrees(r.Z()),
            "error": solution.reprojection_error,
        }

    return {
        "camera": index,
        "calibrated": calibrated,
        "detected_ids": sorted(d.tag_id for d in detections),
        "used_ids": observation.tag_ids if observation else [],
        "too_small_ids": sorted(d.tag_id for d in excluded),
        "search": search.mode,
        "searched_pct": round(100 * search.searched_fraction, 1),
        "detect_ms": round(search.detect_ms, 2),
        "pose": pose(observation.best) if observation else None,
        "alternate_pose": (
            pose(observation.alternate)
            if observation and observation.alternate
            else None
        ),
    }


def _overlay(
    detections: list[list[TagDetection]], searches: list[SearchReport]
) -> Overlay:
    def draw(images: Sequence[numpy.ndarray]) -> None:
        for image, camera_detections, search in zip(images, detections, searches):
            # Where temporal acceleration searched (cyan); nothing drawn for a full search.
            for x0, y0, x1, y1 in search.regions:
                cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 0), 2)
            draw_tag_detections(image, camera_detections)

    return draw


class _FpsCounter:
    """Frames processed in the last full second."""

    def __init__(self) -> None:
        self._window_start = time.monotonic()
        self._count = 0
        self._fps = 0

    def tick(self) -> int:
        self._count += 1
        now = time.monotonic()
        if now - self._window_start >= 1.0:
            self._fps = self._count
            self._count = 0
            self._window_start = now
        return self._fps
