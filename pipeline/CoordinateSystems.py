from typing import List, Tuple
import cv2
import numpy
import numpy.typing
import math

from wpimath.geometry import Pose3d, Rotation3d, Translation3d


def openCvPoseToWpilib(tvec: cv2.typing.MatLike, rvec: cv2.typing.MatLike) -> Pose3d:
    return Pose3d(
        Translation3d(tvec[2][0], -tvec[0][0], -tvec[1][0]),
        Rotation3d(
            numpy.array([rvec[2][0], -rvec[0][0], -rvec[1][0]]),
            math.sqrt(
                math.pow(rvec[0][0], 2)
                + math.pow(rvec[1][0], 2)
                + math.pow(rvec[2][0], 2)
            ),
        ),
    )


def wpilibTranslationToOpenCv(translation: Translation3d) -> List[float]:
    return [-translation.Y(), -translation.Z(), translation.X()]
