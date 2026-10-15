"""Temporal acceleration: only search the parts of the image where tags should be.

From the camera's recent poses, predict where every field tag will appear in
this frame, and run the detector only on padded regions around those spots
instead of the whole image. Regions are usually a small fraction of the
image, so detection is several times faster.

Idea from EagleEye's "Temporal Acceleration" (Scythe Engineering). This is an
independent implementation: EagleEye is PolyForm Noncommercial licensed, which
is incompatible with C2's AGPL. Differences from EagleEye's version:

- Motion prediction: the pose is extrapolated at constant velocity from the
  last two solves to this frame's capture time, so fast driving and turning
  don't outrun the padding. (EagleEye reuses the last pose as-is.)
- Recovery: if the predicted regions yield no tags, the full image is searched
  in the same frame, so losing track never costs a frame. A full search also
  runs every ``full_frame_interval`` frames, to find tags the prediction
  missed, and whenever the pose is stale. (EagleEye keeps searching the old
  regions until the pipeline is reconfigured.)
- Projection uses the full calibrated lens model (cv2.projectPoints).
- Regions are plain crops (views into the image, no resampling), so detected
  corners are exact after adding the crop's offset, and overlapping regions
  are merged so each tag is searched once.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy
from wpimath.geometry import Pose3d, Twist3d

from config.local_config import CameraCalibration
from pipelines.apriltag.detector import ArucoNanoDetector, TagDetection
from pipelines.apriltag.tag_layout import FieldTag

MAX_POSE_AGE_US = 250_000
"""Don't predict from a pose older than this; search the full image instead."""
_MAX_EXTRAPOLATION = 2.0
"""Extrapolate at most this many pose intervals ahead (guards against a stale pair)."""
_MIN_PADDING_PX = 24
"""Padding added around every predicted tag, on top of the proportional padding."""
_MIN_REGION_PX = 16
_NEAR_PLANE_M = 0.05
_FRUSTUM_MARGIN = 1.5
"""Project only corners within this multiple of the image's half-size (beyond it,
the distortion polynomial is meaningless)."""


@dataclass(frozen=True)
class TrackingSettings:
    enabled: bool
    padding: float
    """Region padding, as a fraction of the predicted tag's size."""
    full_frame_interval: int
    """Search the full image at least every N frames. 0 = only when tracking fails."""


@dataclass(frozen=True)
class SearchReport:
    mode: str
    """"full" (whole image), "regions" (predicted regions only), or
    "regions+full" (regions found nothing, so the whole image was searched too)."""
    regions: tuple[tuple[int, int, int, int], ...]
    """Searched regions (x0, y0, x1, y1); empty for a full search."""
    searched_fraction: float
    detect_ms: float


class CameraTracker:
    """Per-camera tracking state. Not thread-safe: one camera, one thread at a time."""

    def __init__(self) -> None:
        self._poses: deque[tuple[int, Pose3d]] = deque(maxlen=2)
        self._frames_since_full = 0
        self.last_report = SearchReport("full", (), 1.0, 0.0)

    def record(self, capture_us: int, field_to_camera: Pose3d | None) -> None:
        """Feed the camera pose solved for the frame captured at ``capture_us`` (None if unsolved)."""
        if field_to_camera is not None:
            self._poses.append((capture_us, field_to_camera))

    def predicted_pose(self, capture_us: int) -> Pose3d | None:
        """Where the camera is expected to be at ``capture_us``, or None if we don't know."""
        if not self._poses:
            return None
        t1, p1 = self._poses[-1]
        if capture_us - t1 > MAX_POSE_AGE_US:
            return None
        if len(self._poses) < 2:
            return p1
        t0, p0 = self._poses[0]
        if t1 <= t0:
            return p1
        scale = min((capture_us - t1) / (t1 - t0), _MAX_EXTRAPOLATION)
        twist = p0.log(p1)
        return p1.exp(
            Twist3d(
                *(
                    scale * v
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

    def detect(
        self,
        detector: ArucoNanoDetector,
        image: numpy.ndarray,
        capture_us: int,
        tags: TagGeometry,
        calibration: CameraCalibration | None,
        settings: TrackingSettings,
    ) -> list[TagDetection]:
        started = time.perf_counter()
        height, width = image.shape[:2]
        pose = (
            self.predicted_pose(capture_us)
            if settings.enabled and calibration is not None
            else None
        )
        due_for_full = 0 < settings.full_frame_interval <= self._frames_since_full

        mode, regions = "full", ()
        if pose is not None and tags.layout and not due_for_full:
            assert calibration is not None
            predicted = predict_regions(
                pose, tags, calibration, (width, height), settings.padding
            )
            regions = tuple(_merge_overlapping(predicted))
            if regions:
                detections = detect_in_regions(detector, image, regions)
                if detections:
                    self._frames_since_full += 1
                    if len(detections) * 2 < len(predicted):
                        # Most predicted tags weren't found (fast motion?): full search next frame.
                        self._frames_since_full = max(
                            self._frames_since_full, settings.full_frame_interval
                        )
                    self._report("regions", regions, (width, height), started)
                    return detections
                mode = "regions+full"  # predicted tags weren't there: search everything, this frame

        detections = detector.detect(image)
        self._frames_since_full = 0
        self._report(
            mode, regions if mode == "regions+full" else (), (width, height), started
        )
        return detections

    def _report(
        self, mode: str, regions, size: tuple[int, int], started: float
    ) -> None:
        area = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in regions)
        fraction = 1.0 if mode != "regions" else area / float(size[0] * size[1])
        self.last_report = SearchReport(
            mode, regions, fraction, (time.perf_counter() - started) * 1000
        )


class TagGeometry:
    """Field tags as arrays, computed once per layout so per-frame prediction is vectorized."""

    def __init__(self, layout: dict[int, FieldTag]) -> None:
        self.layout = layout
        tags = list(layout.values())
        # corners_opencv holds field points in OpenCV axis order (-Y, -Z, X); undo that.
        corners = numpy.array(
            [t.corners_opencv for t in tags], dtype=numpy.float64
        ).reshape(-1, 4, 3)
        self.corners = numpy.stack(
            [corners[..., 2], -corners[..., 0], -corners[..., 1]], axis=-1
        )
        self.centers = numpy.array(
            [[t.pose.X(), t.pose.Y(), t.pose.Z()] for t in tags]
        ).reshape(-1, 3)
        # A tag's +X axis points out of its face.
        self.normals = numpy.array(
            [t.pose.rotation().toMatrix()[:, 0] for t in tags]
        ).reshape(-1, 3)


def predict_regions(
    field_to_camera: Pose3d,
    tags: TagGeometry,
    calibration: CameraCalibration,
    image_size: tuple[int, int],
    padding: float,
) -> list[tuple[int, int, int, int]]:
    """Padded regions (x0, y0, x1, y1), one per tag expected to be visible (not merged)."""
    if len(tags.centers) == 0:
        return []
    width, height = image_size
    k = calibration.camera_matrix
    rotation = field_to_camera.rotation().toMatrix()
    position = numpy.array(
        [field_to_camera.X(), field_to_camera.Y(), field_to_camera.Z()]
    )

    facing = numpy.einsum("ij,ij->i", tags.normals, position - tags.centers) > 0
    in_camera = (tags.corners - position) @ rotation  # WPILib camera axes, (N, 4, 3)
    opencv = numpy.stack(
        [-in_camera[..., 1], -in_camera[..., 2], in_camera[..., 0]], axis=-1
    )
    z = opencv[..., 2]
    with numpy.errstate(divide="ignore", invalid="ignore"):
        slope_x, slope_y = numpy.abs(opencv[..., 0] / z), numpy.abs(opencv[..., 1] / z)
    max_x = width / 2 * _FRUSTUM_MARGIN / k[0, 0]
    max_y = height / 2 * _FRUSTUM_MARGIN / k[1, 1]
    visible = (
        facing
        & (z > _NEAR_PLANE_M).all(axis=1)
        & (slope_x < max_x).all(axis=1)
        & (slope_y < max_y).all(axis=1)
    )
    if not visible.any():
        return []

    points = opencv[visible].reshape(-1, 3)
    pixels = cv2.projectPoints(
        points, numpy.zeros(3), numpy.zeros(3), k, calibration.distortion_coefficients
    )[0]
    pixels = pixels.reshape(-1, 4, 2)
    low, high = pixels.min(axis=1), pixels.max(axis=1)
    center, half = (low + high) / 2, (high - low) / 2
    pad = half.max(axis=1, keepdims=True) * padding + _MIN_PADDING_PX
    x0y0 = numpy.clip(numpy.floor(center - half - pad), 0, [width, height]).astype(int)
    x1y1 = numpy.clip(numpy.ceil(center + half + pad), 0, [width, height]).astype(int)
    return [
        (int(a[0]), int(a[1]), int(b[0]), int(b[1]))
        for a, b in zip(x0y0, x1y1)
        if b[0] - a[0] >= _MIN_REGION_PX and b[1] - a[1] >= _MIN_REGION_PX
    ]


def detect_in_regions(
    detector: ArucoNanoDetector, image: numpy.ndarray, regions
) -> list[TagDetection]:
    found: dict[int, TagDetection] = {}
    for x0, y0, x1, y1 in regions:
        for detection in detector.detect(image[y0:y1, x0:x1]):
            if detection.tag_id not in found:
                found[detection.tag_id] = TagDetection(
                    detection.tag_id, detection.corners + (x0, y0)
                )
    return list(found.values())


def _merge_overlapping(
    boxes: list[tuple[int, int, int, int]],
) -> list[tuple[int, int, int, int]]:
    """Union overlapping boxes until none overlap (so each tag is searched once)."""
    merged = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(merged)):
            for j in range(i + 1, len(merged)):
                a, b = merged[i], merged[j]
                if a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]:
                    merged[i] = (
                        min(a[0], b[0]),
                        min(a[1], b[1]),
                        max(a[2], b[2]),
                        max(a[3], b[3]),
                    )
                    del merged[j]
                    changed = True
                    break
            if changed:
                break
    return merged
