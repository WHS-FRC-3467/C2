"""Field-relative camera pose from AprilTag corners (solvePnP)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy
from wpimath.geometry import Pose3d, Transform3d

from config.local_config import CameraCalibration
from pipelines.apriltag.detector import TagDetection
from pipelines.apriltag.tag_layout import FieldTag
from vision.coordinate_systems import opencv_pose_to_wpilib


@dataclass(frozen=True)
class PoseSolution:
    field_to_camera: Pose3d
    reprojection_error: float


@dataclass(frozen=True)
class CameraPoseObservation:
    tag_ids: list[int]
    best: PoseSolution
    alternate: PoseSolution | None
    """Only for single-tag observations, where the pose is ambiguous."""


def solve_camera_pose(
    detections: list[TagDetection],
    layout: dict[int, FieldTag],
    tag_size_m: float,
    calibration: CameraCalibration,
) -> CameraPoseObservation | None:
    """Solve for the field-to-camera pose. Returns None if no known tags are visible."""
    known = [d for d in detections if d.tag_id in layout]
    if not known:
        return None
    if len(known) == 1:
        return _solve_single_tag(known[0], layout[known[0].tag_id], tag_size_m, calibration)
    return _solve_multi_tag(known, layout, calibration)


def _solve_single_tag(
    detection: TagDetection, tag: FieldTag, tag_size_m: float, calibration: CameraCalibration
) -> CameraPoseObservation | None:
    """IPPE_SQUARE returns both candidate poses of a lone square; report both."""
    half = tag_size_m / 2.0
    # Tag-frame corners in the order SOLVEPNP_IPPE_SQUARE requires.
    object_points = numpy.array(
        [[-half, half, 0.0], [half, half, 0.0], [half, -half, 0.0], [-half, -half, 0.0]]
    )
    try:
        _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            object_points,
            detection.corners,
            calibration.camera_matrix,
            calibration.distortion_coefficients,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
        )
    except cv2.error:
        return None

    solutions = []
    for rvec, tvec, error in zip(rvecs, tvecs, errors):
        camera_to_tag = opencv_pose_to_wpilib(tvec, rvec)
        tag_to_camera = Transform3d(camera_to_tag.translation(), camera_to_tag.rotation()).inverse()
        solutions.append(PoseSolution(tag.pose.transformBy(tag_to_camera), float(error[0])))
    return CameraPoseObservation(
        [detection.tag_id], solutions[0], solutions[1] if len(solutions) > 1 else None
    )


def _solve_multi_tag(
    detections: list[TagDetection], layout: dict[int, FieldTag], calibration: CameraCalibration
) -> CameraPoseObservation | None:
    """SQPNP over every corner of every tag gives one unambiguous pose."""
    object_points = numpy.concatenate([layout[d.tag_id].corners_opencv for d in detections])
    image_points = numpy.concatenate([d.corners for d in detections])
    try:
        _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
            object_points,
            image_points,
            calibration.camera_matrix,
            calibration.distortion_coefficients,
            flags=cv2.SOLVEPNP_SQPNP,
        )
    except cv2.error:
        return None

    # solvePnP gives camera-to-field; invert for field-to-camera.
    camera_to_field = opencv_pose_to_wpilib(tvecs[0], rvecs[0])
    field_to_camera = Transform3d(camera_to_field.translation(), camera_to_field.rotation()).inverse()
    return CameraPoseObservation(
        [d.tag_id for d in detections],
        PoseSolution(Pose3d(field_to_camera.translation(), field_to_camera.rotation()), float(errors[0][0])),
        None,
    )
