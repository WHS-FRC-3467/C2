"""Guided data collection: show the user where to hold the board next.

Adapted from Pose Calib by Pavel Rojtberg (AGPL-3.0, see LICENSE), "Efficient
Pose Selection for Interactive Camera Calibration", ISMAR 2018. After each
captured keyframe the intrinsics are re-estimated, and the next target pose
is chosen to shrink the uncertainty of the least-constrained parameter.

Changes from Pose Calib:
- The image size comes from the camera.
- Stillness is only judged between two *different* frames (Pose Calib could
  compare a frame with itself and capture a blurred, moving board).
- Keyframes whose own reprojection error is far above the others' (motion
  blur, a misdetected corner) are dropped.
- Converged requires a minimum number of keyframes and small absolute
  uncertainty for fx, fy, cx, cy, not only "the last frame didn't help much".
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy

from calibration.board import Board, BoardObservation
from calibration.estimator import INTRINSIC_NAMES, NUM_INTRINSICS, Calibrator, Keyframe
from calibration.posegen import PoseGeneratorDist
from calibration.preview import BoardPreview

# Pose Calib's thresholds
_OVERLAP_TO_CAPTURE = 0.85
_STILL_PIXELS = 2.0
# Parameters optimized by the same kind of board poses
_PARAM_GROUPS = [(0, 1, 2, 3), (4, 5, 6, 7, 8)]

# Stricter stopping (see module docs)
VAR_TERMINATE = 0.05
"""A parameter is converged when a new keyframe shrinks its std dev by less than this fraction."""
MIN_KEYFRAMES = 10
MAX_CORE_STD_PX = 1.0
"""Max std dev of fx, fy, cx, cy before stopping, pixels."""

# Dropping bad keyframes
_BAD_KEYFRAME_FACTOR = 3.0
_BAD_KEYFRAME_MIN_PX = 1.0


@dataclass(frozen=True)
class GuidanceStatus:
    keyframes: int
    converged: bool
    instruction: str
    target_param: str | None
    rms_error_px: float | None
    intrinsics: dict[str, float]
    intrinsics_std: dict[str, float]
    params_converged: dict[str, bool]
    last_event: str


class Guidance:
    def __init__(self, board: Board, image_size: tuple[int, int]) -> None:
        self.board = board
        self.image_size = image_size
        self.calibrator = Calibrator(image_size)
        self.preview = BoardPreview(board)
        self._generator = PoseGeneratorDist(image_size)
        spec = board.spec
        # Target poses are in the board's units (meters); depth is in board widths.
        self._board_extent = (
            numpy.array([spec.squares_x, spec.squares_y, spec.squares_x])
            * spec.square_m
        )
        self._min_init_points = spec.num_corners // 2
        self._best_init_error = float("inf")
        self._params_converged = numpy.zeros(NUM_INTRINSICS, dtype=bool)
        self._target_param: int | None = None
        self._target_mask: numpy.ndarray | None = None
        self.target_image: numpy.ndarray | None = None
        """The board drawn at the target pose (BGR, black where empty), or None when done."""
        self.converged = False
        self._previous: BoardObservation | None = None
        self.last_event = "Hold the board in front of the camera."
        self._set_next_pose()

    @property
    def keyframes(self) -> list[Keyframe]:
        return self.calibrator.keyframes

    def update(
        self, observation: BoardObservation | None, sequence: int, force: bool = False
    ) -> bool:
        """Feed one NEW frame's detection. Returns True if it was captured as a keyframe."""
        moving = _mean_motion(self._previous, observation)
        self._previous = observation
        if observation is None:
            return False

        k, distortion = self.calibrator.k, self.calibrator.distortion

        # Bootstrap the intrinsics from a single, well-filled view before the first keyframe.
        if not self.keyframes and observation.count >= self._min_init_points:
            self.calibrator.calibrate([Keyframe(observation, sequence)])
            if (
                numpy.isfinite(self.calibrator.k).all()
                and self.calibrator.rms_error < self._best_init_error
            ):
                self._best_init_error = self.calibrator.rms_error
                self._set_next_pose()
            k, distortion = self.calibrator.k, self.calibrator.distortion

        at_target = force or self._overlaps_target(observation, k, distortion)
        # Enough points to constrain the unknowns (Pose Calib's rule).
        needed = (
            ((NUM_INTRINSICS + 2 * 6) * 5 + 3) // 4 if len(self.keyframes) < 2 else 15
        )
        if observation.count < needed:
            if at_target:
                self.last_event = f"Need more of the board visible ({observation.count}/{needed} corners)."
            return False
        if not at_target:
            return False
        if not force and (moving is None or moving > _STILL_PIXELS):
            self.last_event = "Hold still..."
            return False

        self.keyframes.append(Keyframe(observation, sequence))
        self._recalibrate()
        self.last_event = f"Captured keyframe {len(self.keyframes)}."
        self._drop_bad_keyframes()
        self.converged = self._is_converged()
        if self.converged:
            self.target_image = None
            self._target_mask = None
            self.last_event = "Enough data collected."
        else:
            self._set_next_pose()
        return True

    def resume(self) -> None:
        """Keep collecting after convergence (e.g. the solve was too uncertain)."""
        self.converged = False
        self._set_next_pose()

    def status(self) -> GuidanceStatus:
        c = self.calibrator
        intrinsics = c.intrinsics()[:NUM_INTRINSICS] if c.distortion is not None else []
        std = c.intrinsics_std()
        return GuidanceStatus(
            keyframes=len(self.keyframes),
            converged=self.converged,
            instruction=self._instruction(),
            target_param=(
                INTRINSIC_NAMES[self._target_param]
                if self._target_param is not None
                else None
            ),
            rms_error_px=float(c.rms_error) if numpy.isfinite(c.rms_error) else None,
            intrinsics={n: float(v) for n, v in zip(INTRINSIC_NAMES, intrinsics)},
            intrinsics_std=(
                {n: float(v) for n, v in zip(INTRINSIC_NAMES, std)}
                if len(self.keyframes) >= 2
                else {}
            ),
            params_converged={
                n: bool(v) for n, v in zip(INTRINSIC_NAMES, self._params_converged)
            },
            last_event=self.last_event,
        )

    # --------------------------------------------------------------- private

    def _recalibrate(self) -> None:
        if len(self.keyframes) < 2:
            return
        c = self.calibrator
        previous = numpy.diag(c.information)[:NUM_INTRINSICS].copy()
        first = len(self.keyframes) == 2
        dispersion = c.calibrate()
        current = numpy.diag(c.information)[:NUM_INTRINSICS]
        if not first and self._target_param is not None:
            with numpy.errstate(divide="ignore", invalid="ignore"):
                relative_gain = 1 - numpy.sqrt(current) / numpy.sqrt(previous)
            for group in _PARAM_GROUPS:
                if self._target_param in group:
                    for p in group:
                        if 0 < relative_gain[p] < VAR_TERMINATE:
                            self._params_converged[p] = True
        dispersion = dispersion.copy()
        dispersion[self._params_converged] = 0
        self._target_param = int(dispersion.argmax())

    def _drop_bad_keyframes(self) -> None:
        errors = self.calibrator.keyframe_errors
        if len(errors) < 4:
            return
        limit = max(
            _BAD_KEYFRAME_FACTOR * float(numpy.median(errors)), _BAD_KEYFRAME_MIN_PX
        )
        bad = [i for i, e in enumerate(errors) if e > limit]
        if not bad:
            return
        for i in reversed(bad):
            del self.keyframes[i]
        self._recalibrate()
        self.last_event = f"Dropped {len(bad)} bad keyframe(s) (blur or misdetection)."

    def _is_converged(self) -> bool:
        if len(self.keyframes) < MIN_KEYFRAMES or not self._params_converged.all():
            return False
        return bool((self.calibrator.intrinsics_std()[:4] <= MAX_CORE_STD_PX).all())

    def _set_next_pose(self) -> None:
        c = self.calibrator
        target = self._target_param if self._target_param is not None else 0
        rvec, tvec = self._generator.get_pose(
            self._board_extent, len(self.keyframes), target, c.k, c.distortion
        )
        rvec, tvec = _from_legacy_board_frame(rvec, tvec, self._board_extent[1])
        self.preview.set_camera(c.k, c.distortion, self.image_size)
        self._target_pose = (rvec, tvec)
        self.target_image = self.preview.project(rvec, tvec)
        self._target_mask = self.preview.project(rvec, tvec, mask=True)

    def _overlaps_target(self, observation: BoardObservation, k, distortion) -> bool:
        """Jaccard overlap between the board where it is and where we want it."""
        if self._target_mask is None:
            return False
        pose = self.board.estimate_pose(observation, k, distortion)
        if pose is None:
            return False
        current = self.preview.project(*pose, mask=True)
        target = self._target_mask
        intersection = numpy.count_nonzero(current & target)
        union = numpy.count_nonzero(current | target)
        return union > 0 and intersection / union > _OVERLAP_TO_CAPTURE

    def _instruction(self) -> str:
        if self.converged:
            return "Done collecting. Solving with mrcal..."
        if len(self.keyframes) < 2:
            return "Match the green board with the real one, then hold still."
        assert self._target_param is not None
        variance = self.calibrator.pose_variance
        if self._target_param < 2:
            axis = ("tilt up/down", "tilt left/right")[int(variance[:2].argmin())]
        else:
            axis = ("move left/right", "move up/down", "move closer/farther")[
                int(variance[3:6].argmin())
            ]
        return f"Match the green board ({axis}) to improve {INTRINSIC_NAMES[self._target_param]}."


def _from_legacy_board_frame(
    rvec, tvec, board_height_m: float
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Convert a pose of the board from OpenCV's old ChArUco frame to the current one.

    posegen (from Pose Calib) generates poses for OpenCV's pre-4.6 board frame:
    origin at the bottom-left corner, y up. The current frame has its origin at
    the top-left, y down. A point is p_old = Rx(pi) p_new + (0, height, 0), so
    camera = R p_old + t = (R Rx(pi)) p_new + (t + R (0, height, 0)).
    Without this the target board is shown from behind (mirrored).
    """
    rotation = cv2.Rodrigues(numpy.asarray(rvec, dtype=numpy.float64))[0]
    flip = numpy.diag([1.0, -1.0, -1.0])  # Rx(pi)
    new_rotation = rotation @ flip
    new_tvec = numpy.asarray(tvec, dtype=numpy.float64) + rotation @ numpy.array(
        [0.0, board_height_m, 0.0]
    )
    return cv2.Rodrigues(new_rotation)[0].ravel(), new_tvec


def _mean_motion(
    previous: BoardObservation | None, current: BoardObservation | None
) -> float | None:
    """Mean pixel motion of corners seen in both frames, or None if not comparable."""
    if previous is None or current is None:
        return None
    common, i_prev, i_cur = numpy.intersect1d(
        previous.corner_ids, current.corner_ids, return_indices=True
    )
    if len(common) < 4:
        return None
    return float(
        numpy.linalg.norm(
            previous.image_points[i_prev] - current.image_points[i_cur], axis=1
        ).mean()
    )
