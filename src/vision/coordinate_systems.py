"""Conversions between OpenCV's camera frame and WPILib's coordinate system.

OpenCV camera frame: +X right, +Y down, +Z forward (out of the lens).
WPILib:              +X forward, +Y left, +Z up.
"""

import math

import numpy
from wpimath.geometry import Pose3d, Rotation3d, Translation3d


def opencv_pose_to_wpilib(tvec: numpy.ndarray, rvec: numpy.ndarray) -> Pose3d:
    """Convert a solvePnP result (tvec, Rodrigues rvec) to a WPILib Pose3d."""
    return Pose3d(
        Translation3d(tvec[2][0], -tvec[0][0], -tvec[1][0]),
        Rotation3d(
            numpy.array([rvec[2][0], -rvec[0][0], -rvec[1][0]]),
            math.sqrt(rvec[0][0] ** 2 + rvec[1][0] ** 2 + rvec[2][0] ** 2),
        ),
    )


def wpilib_translation_to_opencv(translation: Translation3d) -> list[float]:
    return [-translation.Y(), -translation.Z(), translation.X()]
