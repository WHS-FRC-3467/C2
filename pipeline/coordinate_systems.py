from typing import List, Tuple
import numpy
from wpimath.geometry import *
import numpy.typing
import math


def openCvPoseToWpilib(tvec: numpy.typing.NDArray[numpy.float64], rvec: numpy.typing.NDArray[numpy.float64]) -> Pose3d:
    return Pose3d(
        Translation3d(tvec[2][0], -tvec[0][0], -tvec[1][0]),
        Rotation3d(
            numpy.array([rvec[2][0], -rvec[0][0], -rvec[1][0]]),
            math.sqrt(math.pow(rvec[0][0], 2) + math.pow(rvec[1][0], 2) + math.pow(rvec[2][0], 2))
        ))


def wpilibTranslationToOpenCv(translation: Translation3d) -> List[float]:
    return [-translation.Y(), -translation.Z(), translation.X()]


def wpilibPoseToOpenCv(pose: Pose3d) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """Convert a WPILib Pose3d to OpenCV rvec/tvec."""
    t = pose.translation()
    tvec = numpy.array([[-t.Y()], [-t.Z()], [t.X()]])
    r = pose.rotation()
    axis = r.axis()
    angle = r.angle()
    rvec = numpy.array([[-axis.y * angle], [-axis.z * angle], [axis.x * angle]])
    return rvec, tvec
