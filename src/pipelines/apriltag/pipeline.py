"""AprilTag pipeline: per-camera field-relative pose from AprilTags.

Inputs:  every camera of the multi-camera module.
Outputs: /<dev>/pipelines/apriltag/output/camera_<i>   (schema/apriltag.fbs, "dsv0_fb")
         One message per camera per frame, including frames with no tags,
         so the robot can tell "no tags seen" apart from "disconnected".
Config:  /<dev>/pipelines/apriltag/config/tag_layout       AprilTagFieldLayout JSON
         /<dev>/pipelines/apriltag/config/fiducial_size_m  tag edge length, meters
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy

from pipelines.apriltag.detector import ArucoNanoDetector, TagDetection
from pipelines.apriltag.encoder import TYPE_STRING, encode_camera_frame
from pipelines.apriltag.pose_solver import solve_camera_pose
from pipelines.apriltag.tag_layout import FieldTag, parse_tag_layout
from pipelines.pipeline import Overlay, Pipeline, PipelineContext, PipelineInputs
from vision.overlay import draw_tag_detections

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class AprilTagSettings:
    tag_layout: str = ""
    fiducial_size_m: float = 0.1651  # 6.5 in


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
        self._layout: dict[int, FieldTag] = {}
        self._tag_size_m = 0.0
        self._fps = _FpsCounter()

    def configure(self, settings: AprilTagSettings) -> None:
        self._tag_size_m = settings.fiducial_size_m
        if not settings.tag_layout:
            self._layout = {}
            log.warning("No tag layout yet; publishing empty results")
            return
        try:
            self._layout = parse_tag_layout(settings.tag_layout, settings.fiducial_size_m)
            log.info("Loaded tag layout with %d tags", len(self._layout))
        except ValueError:
            self._layout = {}
            log.exception("Ignoring tag layout; publishing empty results")

    def process(self, inputs: PipelineInputs) -> Overlay | None:
        frames = inputs.frames
        assert frames is not None
        fps = self._fps.tick()

        detections = list(
            self._pool.map(lambda i: self._detectors[i].detect(frames.cameras[i]), range(frames.num_cameras))
        )

        # None if we aren't synced with the robot: skip solving, nothing can be sent.
        stamp = self.context.time_sync.stamp(frames.capture_time_us)
        if stamp is not None:
            for i, camera_detections in enumerate(detections):
                calibration = self.context.calibrations[i]
                observation = None
                if calibration is not None and self._layout:
                    observation = solve_camera_pose(camera_detections, self._layout, self._tag_size_m, calibration)
                self._outputs[i].publish(encode_camera_frame(i, observation, fps), stamp)

        return _overlay(detections)

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        for detector in self._detectors:
            detector.close()


def _overlay(detections: list[list[TagDetection]]) -> Overlay:
    def draw(images: Sequence[numpy.ndarray]) -> None:
        for image, camera_detections in zip(images, detections):
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
