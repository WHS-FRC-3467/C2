from dataclasses import dataclass
from typing import List, Union

import numpy
import numpy.typing
from wpimath.geometry import *


@dataclass(frozen=True)
class FiducialImageObservation:
    tag_id: int
    corners: numpy.typing.NDArray[numpy.float64]


@dataclass(frozen=True)
class CameraPoseObservation:
    tag_ids: List[int]
    pose_0: Pose3d
    error_0: float
    pose_1: Union[Pose3d, None]
    error_1: Union[float, None]


@dataclass(frozen=True)
class ObjectDetectionObservation:
    class_id: int
    confidence: float
    x0: int
    y0: int
    x1: int
    y1: int
    centroid_x: float
    centroid_y: float
    area_px: int
    pitch_deg: float
    yaw_deg: float
