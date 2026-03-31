import math

import numpy
from cv2.typing import MatLike
from wpimath.geometry import Pose3d, Rotation3d, Translation3d  # type: ignore[import-not-found]


def openCvPoseToWpilib(tvec: MatLike, rvec: MatLike) -> Pose3d:
    tvec_array = numpy.asarray(tvec, dtype=numpy.float64).reshape(3, 1)
    rvec_array = numpy.asarray(rvec, dtype=numpy.float64).reshape(3, 1)
    return Pose3d(
        Translation3d(tvec_array[2][0], -tvec_array[0][0], -tvec_array[1][0]),
        Rotation3d(
            numpy.array([rvec_array[2][0], -rvec_array[0][0], -rvec_array[1][0]]),
            math.sqrt(
                float(
                    rvec_array[0][0] ** 2
                    + rvec_array[1][0] ** 2
                    + rvec_array[2][0] ** 2
                )
            ),
        ),
    )


def wpilibTranslationToOpenCv(translation: Translation3d) -> list[float]:
    return [-translation.Y(), -translation.Z(), translation.X()]
