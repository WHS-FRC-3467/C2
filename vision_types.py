from dataclasses import dataclass

from wpimath.geometry import Pose3d  # type: ignore[import-not-found]

from type_defs import FloatArray


@dataclass(frozen=True)
class FiducialImageObservation:
    tag_id: int
    corners: FloatArray


@dataclass(frozen=True)
class FiducialPoseObservation:
    tag_id: int
    pose_0: Pose3d
    error_0: float
    tvec_0: FloatArray
    rvec_0: FloatArray
    pose_1: Pose3d | None
    error_1: float | None
    tvec_1: FloatArray | None
    rvec_1: FloatArray | None


@dataclass(frozen=True)
class CameraPoseObservation:
    tag_ids: list[int]
    pose_0: Pose3d
    error_0: float
    pose_1: Pose3d | None
    error_1: float | None


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
