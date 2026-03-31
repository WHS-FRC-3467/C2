import cv2
import numpy
from config.config import ConfigStore
from vision_types import FiducialImageObservation, FiducialPoseObservation

from pipeline.coordinate_systems import openCvPoseToWpilib


class PoseEstimator:
    def __init__(self) -> None:
        raise NotImplementedError

    def solve_fiducial_pose(
        self, image_observation: FiducialImageObservation, config_store: ConfigStore
    ) -> FiducialPoseObservation | None:
        raise NotImplementedError


class SquareTargetPoseEstimator(PoseEstimator):
    def __init__(self) -> None:
        return None

    def solve_fiducial_pose(
        self, image_observation: FiducialImageObservation, config_store: ConfigStore
    ) -> FiducialPoseObservation | None:
        fid_size = config_store.remote_config.fiducial_size_m
        if fid_size <= 0:
            return None

        object_points = numpy.array(
            [
                [-fid_size / 2.0, fid_size / 2.0, 0.0],
                [fid_size / 2.0, fid_size / 2.0, 0.0],
                [fid_size / 2.0, -fid_size / 2.0, 0.0],
                [-fid_size / 2.0, -fid_size / 2.0, 0.0],
            ],
            dtype=numpy.float64,
        )

        try:
            _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                object_points,
                image_observation.corners,
                config_store.local_config.camera_matrix,
                config_store.local_config.distortion_coefficients,
                flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
        except cv2.error:
            return None

        primary_rvec = numpy.asarray(rvecs[0], dtype=numpy.float64)
        primary_tvec = numpy.asarray(tvecs[0], dtype=numpy.float64)
        primary_error = float(errors[0][0])

        secondary_pose = None
        secondary_error = None
        secondary_tvec = None
        secondary_rvec = None
        if len(rvecs) > 1 and len(tvecs) > 1 and len(errors) > 1:
            secondary_rvec = numpy.asarray(rvecs[1], dtype=numpy.float64)
            secondary_tvec = numpy.asarray(tvecs[1], dtype=numpy.float64)
            secondary_error = float(errors[1][0])
            secondary_pose = openCvPoseToWpilib(secondary_tvec, secondary_rvec)

        return FiducialPoseObservation(
            tag_id=image_observation.tag_id,
            pose_0=openCvPoseToWpilib(primary_tvec, primary_rvec),
            error_0=primary_error,
            tvec_0=primary_tvec,
            rvec_0=primary_rvec,
            pose_1=secondary_pose,
            error_1=secondary_error,
            tvec_1=secondary_tvec,
            rvec_1=secondary_rvec,
        )
