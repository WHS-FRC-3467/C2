"""Live intrinsics estimate used to guide the user (OpenCV, fast).

Adapted from Pose Calib by Pavel Rojtberg (AGPL-3.0, see LICENSE), "Efficient
Pose Selection for Interactive Camera Calibration", ISMAR 2018.

This only steers data collection. The final calibration is solved by mrcal
(see mrcal_solver.py).

Changes from Pose Calib:
- Image size comes from the camera's frames, not a config file.
- Standard deviations are reported in real units (pixels for fx, fy, cx, cy):
  the covariance is scaled by the residual variance. The convergence test
  still uses the unscaled covariance (pure information gain), as Pose Calib does.
- Per-keyframe reprojection errors are reported, so bad keyframes can be dropped.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy
import numpy.linalg as la

from calibration.board import BoardObservation

NUM_INTRINSICS = 9  # fx, fy, cx, cy, k1, k2, p1, p2, k3
INTRINSIC_NAMES = ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3")


@dataclass(frozen=True)
class Keyframe:
    observation: BoardObservation
    sequence: int


class Calibrator:
    def __init__(self, image_size: tuple[int, int]) -> None:
        self.image_size = image_size
        # Initial guess: square pixels, principal point at the center, empirical focal length.
        self.initial_k = cv2.getDefaultNewCameraMatrix(
            numpy.diag([1000.0, 1000.0, 1.0]), image_size, True
        )
        self.k = self.initial_k.copy()
        self.distortion: numpy.ndarray | None = None
        self.flags = cv2.CALIB_USE_LU

        self.keyframes: list[Keyframe] = []
        self.rms_error = float("nan")
        self.keyframe_errors: list[float] = []
        self.covariance = numpy.zeros((NUM_INTRINSICS, NUM_INTRINSICS))
        """Scaled by the residual variance: real units."""
        self.information = numpy.zeros((NUM_INTRINSICS, NUM_INTRINSICS))
        """Unscaled (J^T J)^-1: how well the data constrains each parameter."""
        self.pose_variance = numpy.zeros(6)

    def intrinsics(self) -> list[float]:
        k = self.k
        distortion = [] if self.distortion is None else list(self.distortion.ravel())
        return [k[0, 0], k[1, 1], k[0, 2], k[1, 2], *distortion]

    def intrinsics_std(self) -> numpy.ndarray:
        """Standard deviation of each intrinsic (fx, fy, cx, cy in pixels)."""
        return numpy.sqrt(
            numpy.maximum(numpy.diag(self.covariance)[:NUM_INTRINSICS], 0)
        )

    def calibrate(self, keyframes: list[Keyframe] | None = None) -> numpy.ndarray:
        """Solve from ``keyframes`` (default: all). Returns the index of dispersion per intrinsic."""
        keyframes = keyframes if keyframes is not None else self.keyframes
        assert keyframes
        flags = self.flags
        if len(keyframes) <= 1:
            # With one view, only the focal length is observable.
            flags |= cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT
            flags |= (
                cv2.CALIB_ZERO_TANGENT_DIST
                | cv2.CALIB_FIX_K1
                | cv2.CALIB_FIX_K2
                | cv2.CALIB_FIX_K3
            )

        object_points = [kf.observation.object_points for kf in keyframes]
        image_points = [kf.observation.image_points for kf in keyframes]
        rms, k, distortion, rvecs, tvecs = cv2.calibrateCamera(
            object_points,
            image_points,
            self.image_size,
            self.initial_k.copy(),
            numpy.zeros((1, 5)),
            flags=flags,
        )
        self.rms_error, self.k, self.distortion = rms, k, distortion
        self.keyframe_errors = [
            _rms_reprojection(op, ip, r, t, k, distortion)
            for op, ip, r, t in zip(object_points, image_points, rvecs, tvecs)
        ]
        self.pose_variance = _pose_variance(rvecs, tvecs)
        self.information = _state_covariance(
            object_points, rvecs, tvecs, k, distortion, flags
        )
        self.covariance = self.information * rms**2
        return _index_of_dispersion(
            self.intrinsics()[:NUM_INTRINSICS],
            numpy.diag(self.information)[:NUM_INTRINSICS],
        )


def _rms_reprojection(object_points, image_points, rvec, tvec, k, distortion) -> float:
    projected = cv2.projectPoints(object_points, rvec, tvec, k, distortion)[0].reshape(
        -1, 2
    )
    return float(
        numpy.sqrt(numpy.mean(numpy.sum((projected - image_points) ** 2, axis=1)))
    )


def _index_of_dispersion(mean, variance) -> numpy.ndarray:
    """https://en.wikipedia.org/wiki/Index_of_dispersion"""
    return numpy.asarray(variance) / [abs(v) if abs(v) > 0 else 1.0 for v in mean]


def _pose_variance(rvecs, tvecs) -> numpy.ndarray:
    euler = numpy.array([cv2.RQDecomp3x3(cv2.Rodrigues(r)[0])[0] for r in rvecs])
    euler[:, 0] = euler[:, 0] % 360  # keep r_x from oscillating between +-180 deg
    variance = numpy.empty(6)
    variance[0:3] = numpy.var(euler, axis=0)
    variance[3:6] = numpy.var(numpy.array(tvecs).reshape(-1, 3), axis=0)
    return variance


def _jacobian_to_multiview(
    jc, points_per_view, num_intrinsics=NUM_INTRINSICS
) -> numpy.ndarray:
    """Expand a compact single-view Jacobian into the sparse multi-view one."""
    total = int(numpy.sum(points_per_view))
    j = numpy.zeros((total * 2, num_intrinsics + 6 * len(points_per_view)))
    j[:, :num_intrinsics] = jc[:, 6 : 6 + num_intrinsics]
    row = 0
    for view, n in enumerate(points_per_view):
        j[
            2 * row : 2 * (row + n),
            num_intrinsics + 6 * view : num_intrinsics + 6 * (view + 1),
        ] = jc[2 * row : 2 * (row + n), :6]
        row += n
    return j


def _state_covariance(
    object_points, rvecs, tvecs, k, distortion, flags
) -> numpy.ndarray:
    """Unscaled covariance (J^T J)^-1 of intrinsics + extrinsics; multiply by the residual variance."""
    if flags & (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3):
        # Fixed parameters make J^T J singular; with one view we don't need it.
        n = NUM_INTRINSICS + 6 * len(object_points)
        return numpy.zeros((n, n))
    points_camera = []
    for points, rvec, tvec in zip(object_points, rvecs, tvecs):
        rotation = cv2.Rodrigues(rvec)[0]
        points_camera.extend(rotation @ p + tvec.ravel() for p in points)
    zero = numpy.zeros(3)
    jc = cv2.projectPoints(
        numpy.array(points_camera), zero, zero, k, distortion[:, :5]
    )[1]
    j = _jacobian_to_multiview(jc, [len(p) for p in object_points])
    return la.pinv(j.T @ j)
