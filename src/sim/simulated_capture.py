"""A stand-in for MultiCameraCapture that renders the field instead of reading a camera.

Run with ``python src/main.py --sim``. Everything downstream (pipelines,
timestamps, NT output, dashboard) is unchanged.

Inputs, from the robot simulation over NT:

    /<dev>/sim/true_pose        Pose3d struct     the robot's true field pose, published every loop
    /<dev>/sim/robot_to_camera  Transform3d[]     where each camera is mounted, indexed by camera
    /<dev>/pipelines/apriltag/config/tag_layout, fiducial_size_m   the field (already sent for the pipeline)

Imperfections, at /<dev>/sim/config/<field> (see SimSettings), live-tunable.
They act on the image and on the hidden "true" hardware, never on the output,
so the real detector and solver react to them exactly as they would on a robot.

Timing: each frame is captured ``pose_lag_ms`` in the past, so the robot's
true pose at that instant has already arrived and can be interpolated
exactly, and never later than the newest true pose received (frames are
never rendered from a guessed pose). The frame is stamped with that capture
time, so robot code can compare vision results against its true pose at the
same timestamp. If the robot sim stops publishing, frames stop too.
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import ntcore
import numpy
from wpimath.geometry import Pose3d, Rotation3d, Transform3d, Translation3d

from capture.multi_camera_capture import CaptureSettings
from config.local_config import CameraCalibration
from core.frames import FrameSet
from core.nt_settings import NTSettings
from core.time_sync import TimeSync, nt_now_us
from pipelines.apriltag.tag_layout import FieldTag, parse_tag_layout
from sim.pose_history import PoseHistory
from sim.renderer import CameraModel, TagRenderer

log = logging.getLogger(__name__)

_NOISE_BANK_SIZE = 8


@dataclass(frozen=True)
class SimSettings:
    """Simulation knobs, at /<dev>/sim/config/<field>. Defaults are a decent camera."""

    fps: float = 45.0
    pose_lag_ms: float = 50.0
    """How far in the past frames are captured, so the robot's true pose at that
    instant is known. Must exceed robot loop period + NT latency (~25 ms)."""

    # Image quality
    image_noise_std: float = 2.0
    """Sensor noise, gray levels (0-255)."""
    blur_sigma_px: float = 0.0
    """Defocus blur."""
    exposure_ms: float = 0.0
    """Motion blur: frames are averaged over this exposure window. 0 = off."""
    motion_blur_samples: int = 4
    lens_distortion: bool = True
    """Render through each camera's calibrated distortion."""

    # Systematic errors: the "true" hardware differs from what C2 and the robot believe.
    mount_error_m: float = 0.0
    """Std dev of each camera's true mounting position error, per axis."""
    mount_error_deg: float = 0.0
    """Std dev of each camera's true mounting angle error, per axis."""
    focal_error_percent: float = 0.0
    """Std dev of each camera's true focal length vs. its calibration."""
    error_seed: int = 1
    """Seed for the fixed per-camera errors above. Change it to draw a new "robot"."""

    # Timing
    extra_latency_ms: float = 0.0
    """Delay before a frame is delivered (like USB transfer). Timestamps stay exact."""
    drop_probability: float = 0.0
    timestamp_error_ms: float = 0.0
    """Constant error added to every frame's timestamp, to test sensitivity to bad timestamps."""
    timestamp_jitter_ms: float = 0.0
    """Random (Gaussian) error added to each frame's timestamp."""


class SimulatedCapture:
    """Same interface as MultiCameraCapture: apply_settings(), read(), snapshot()."""

    def __init__(
        self,
        inst: ntcore.NetworkTableInstance,
        device_id: str,
        num_cameras: int,
        calibrations: tuple[CameraCalibration | None, ...],
        time_sync: TimeSync,
    ) -> None:
        self._num_cameras = num_cameras
        self._calibrations = calibrations
        self._time_sync = time_sync
        self._settings = NTSettings(inst, f"/{device_id}/sim/config", SimSettings)
        self._capture_settings: CaptureSettings | None = None

        sim_table = inst.getTable(f"/{device_id}/sim")
        # periodic: NT batches values to subscribers every 100 ms by default,
        # which would deliver true poses up to 100 ms late.
        # keepDuplicates: a stationary robot publishes the same pose every loop;
        # without it NT drops the repeats and the pose looks stale.
        self._true_pose_sub = sim_table.getStructTopic("true_pose", Pose3d).subscribe(
            Pose3d(),
            ntcore.PubSubOptions(
                sendAll=True, keepDuplicates=True, pollStorage=256, periodic=0.005
            ),
        )
        self._robot_to_camera_sub = sim_table.getStructArrayTopic(
            "robot_to_camera", Transform3d
        ).subscribe([])
        apriltag_config = inst.getTable(f"/{device_id}/pipelines/apriltag/config")
        self._layout_sub = apriltag_config.getStringTopic("tag_layout").subscribe("")
        self._tag_size_sub = apriltag_config.getDoubleTopic(
            "fiducial_size_m"
        ).subscribe(0.1651)

        self._poses = PoseHistory()
        self._renderer = TagRenderer()
        self._pool = ThreadPoolExecutor(
            max_workers=num_cameras, thread_name_prefix="sim-render"
        )
        self._rng = numpy.random.default_rng()
        self._layout_json = ""
        self._layout: dict[int, FieldTag] = {}
        self._sequence = 0
        self._last_capture_us = 0
        self._next_frame_s = time.monotonic()
        self._frame_size: tuple[int, int] | None = None
        self._warnings: list[str] = []
        self._noise_bank: numpy.ndarray | None = None

    # ------------------------------------------------- MultiCameraCapture API

    def apply_settings(self, settings: CaptureSettings) -> None:
        self._capture_settings = settings

    def read(self) -> FrameSet | None:
        settings = self._settings.poll() or self._settings.current
        size = self._camera_size()
        if size is None:
            return None
        while True:
            self._wait_for_next_frame(settings.fps)
            self._update_inputs()
            capture_us = nt_now_us() - round(settings.pose_lag_ms * 1000)
            latest_pose_us = self._poses.latest_time_us
            if latest_pose_us is not None:
                # Never render a pose we don't know yet (and never repeat a timestamp).
                capture_us = min(capture_us, latest_pose_us)
                if capture_us <= self._last_capture_us:
                    continue
            # A dropped frame is simply never delivered, like a real camera's.
            if (
                settings.drop_probability > 0
                and self._rng.random() < settings.drop_probability
            ):
                continue
            break
        self._last_capture_us = capture_us

        images = list(
            self._pool.map(
                lambda i: self._render_camera(i, capture_us, size, settings),
                range(self._num_cameras),
            )
        )
        if settings.extra_latency_ms > 0:
            time.sleep(settings.extra_latency_ms / 1000)

        stamped_us = capture_us + round(
            1000
            * (
                settings.timestamp_error_ms
                + settings.timestamp_jitter_ms * self._rng.standard_normal()
            )
        )
        self._sequence += 1
        combined = numpy.hstack(images)
        self._frame_size = (combined.shape[1], combined.shape[0])
        return FrameSet.from_combined(
            self._sequence, stamped_us, combined, self._num_cameras
        )

    def snapshot(self) -> dict:
        return {
            "simulated": True,
            "open": True,
            "settings": (
                dataclasses.asdict(self._capture_settings)
                if self._capture_settings
                else None
            ),
            "combined_frame_size": self._frame_size,
            "sim_settings": dataclasses.asdict(self._settings.current),
            "warnings": list(self._warnings),
        }

    # ---------------------------------------------------------------- private

    def _camera_size(self) -> tuple[int, int] | None:
        s = self._capture_settings
        if (
            s is None
            or s.camera_resolution_width <= 0
            or s.camera_resolution_height <= 0
        ):
            return None
        return s.camera_resolution_width, s.camera_resolution_height

    def _wait_for_next_frame(self, fps: float) -> None:
        period = 1.0 / max(fps, 1.0)
        now = time.monotonic()
        if self._next_frame_s > now:
            time.sleep(self._next_frame_s - now)
        # If rendering can't keep up, run as fast as possible instead of bursting.
        self._next_frame_s = max(self._next_frame_s + period, time.monotonic())

    def _update_inputs(self) -> None:
        for value in self._true_pose_sub.readQueue():
            self._poses.add(
                self._time_sync.received_to_local_us(value.time), value.value
            )
        layout_json = self._layout_sub.get()
        if layout_json != self._layout_json:
            self._layout_json = layout_json
            try:
                # Only the tag poses are used; the renderer sizes tags itself.
                self._layout = parse_tag_layout(layout_json, 0.0) if layout_json else {}
            except ValueError:
                log.exception("Simulation: bad tag layout")
                self._layout = {}

        warnings = []
        if self._poses.latest_time_us is None:
            warnings.append(
                "No true pose yet: publish Pose3d to /<dev>/sim/true_pose from the robot sim."
            )
        elif nt_now_us() - self._poses.latest_time_us > 500_000:
            warnings.append(
                "True pose is stale (>0.5 s old): is the robot sim running?"
            )
        if len(self._robot_to_camera_sub.get()) < self._num_cameras:
            warnings.append(
                f"Need {self._num_cameras} camera mounts at /<dev>/sim/robot_to_camera."
            )
        if not self._layout:
            warnings.append(
                "No tag layout yet (/<dev>/pipelines/apriltag/config/tag_layout)."
            )
        if warnings != self._warnings:
            for w in warnings:
                log.warning("Simulation: %s", w)
        self._warnings = warnings

    def _render_camera(
        self, index: int, capture_us: int, size: tuple[int, int], s: SimSettings
    ) -> numpy.ndarray:
        width, height = size
        camera = self._true_camera_model(index, width, height, s)
        mounts = self._robot_to_camera_sub.get()
        tag_size = self._tag_size_sub.get()

        # Motion blur: average renders across the exposure window, centered on the capture time.
        samples = max(1, s.motion_blur_samples) if s.exposure_ms > 0 else 1
        offsets_us = (
            [0.0]
            if samples == 1
            else numpy.linspace(-0.5, 0.5, samples) * s.exposure_ms * 1000
        )
        accumulator = None
        for offset in offsets_us:
            robot = self._poses.sample(capture_us + round(offset))
            if robot is None or index >= len(mounts) or not self._layout:
                image = self._renderer.render(
                    camera, Pose3d(), {}, tag_size, background_seed=index
                )
            else:
                field_to_camera = robot.transformBy(mounts[index]).transformBy(
                    self._mount_error(index, s)
                )
                image = self._renderer.render(
                    camera, field_to_camera, self._layout, tag_size, index
                )
            if samples == 1:
                accumulator = image.astype(numpy.float32)
                break
            accumulator = (
                image.astype(numpy.float32)
                if accumulator is None
                else accumulator + image
            )
        assert accumulator is not None
        image = accumulator / len(offsets_us)

        if s.blur_sigma_px > 0:
            image = cv2.GaussianBlur(image, (0, 0), s.blur_sigma_px)
        if s.image_noise_std > 0:
            image += self._noise(width, height) * s.image_noise_std
        return numpy.clip(image, 0, 255).astype(numpy.uint8)

    def _noise(self, width: int, height: int) -> numpy.ndarray:
        """Unit Gaussian noise for one frame.

        Drawn from a small precomputed bank, shifted by a random offset each
        frame: statistically the same as fresh noise, at a tenth of the cost.
        """
        if self._noise_bank is None or self._noise_bank.shape[1:] != (height, width):
            self._noise_bank = self._rng.standard_normal(
                (_NOISE_BANK_SIZE, height, width), dtype=numpy.float32
            )
        frame = self._noise_bank[self._rng.integers(_NOISE_BANK_SIZE)]
        return numpy.roll(
            frame, (self._rng.integers(height), self._rng.integers(width)), axis=(0, 1)
        )

    def _true_camera_model(
        self, index: int, width: int, height: int, s: SimSettings
    ) -> CameraModel:
        """The camera as it really is: its calibration, plus the simulated calibration error."""
        calibration = (
            self._calibrations[index] if index < len(self._calibrations) else None
        )
        if calibration is not None:
            k = calibration.camera_matrix.astype(numpy.float64).copy()
            distortion = calibration.distortion_coefficients.astype(numpy.float64)
        else:
            k, distortion = default_intrinsics(width, height), numpy.zeros((1, 5))
        if s.focal_error_percent:
            scale = (
                1 + s.focal_error_percent / 100 * self._fixed_error(index, "focal")[0]
            )
            k[0, 0] *= scale
            k[1, 1] *= scale
        if not s.lens_distortion:
            distortion = numpy.zeros_like(distortion)
        return CameraModel(width, height, k, distortion)

    def _mount_error(self, index: int, s: SimSettings) -> Transform3d:
        if not s.mount_error_m and not s.mount_error_deg:
            return Transform3d()
        t = self._fixed_error(index, "mount_translation") * s.mount_error_m
        r = numpy.radians(
            self._fixed_error(index, "mount_rotation") * s.mount_error_deg
        )
        return Transform3d(Translation3d(*t), Rotation3d(*r))

    def _fixed_error(self, index: int, kind: str) -> numpy.ndarray:
        """Standard-normal draws that stay fixed for a given camera, kind, and error_seed."""
        seed = (self._settings.current.error_seed, index, sum(map(ord, kind)))
        return numpy.random.default_rng(seed).standard_normal(3)


def default_intrinsics(
    width: int, height: int, horizontal_fov_deg: float = 70.0
) -> numpy.ndarray:
    """A plausible pinhole camera for cameras without a calibration file."""
    focal = (width / 2) / math.tan(math.radians(horizontal_fov_deg) / 2)
    return numpy.array(
        [[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=numpy.float64
    )
