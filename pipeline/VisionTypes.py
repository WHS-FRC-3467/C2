from dataclasses import dataclass
from typing import List, Union

import cv2
from wpimath.geometry import Pose3d


@dataclass(frozen=True)
class FiducialImageObservation:
    tag_id: int
    corners: cv2.typing.MatLike


@dataclass(frozen=True)
class CameraPoseObservation:
    tag_ids: List[int]
    pose_0: Pose3d
    error_0: float
    pose_1: Union[Pose3d, None]
    error_1: Union[float, None]
