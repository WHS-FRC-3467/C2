import cv2
from config.ConfigSource import ConfigSource, FileConfigSource
from config.config import ConfigStore, LocalConfig, RemoteConfig
from pipeline.FiducialDetector import ArucoFiducialDetector
from pipeline.PoseEstimator import SquareTargetPoseEstimator
from wpimath.geometry import Pose3d, Quaternion, Rotation3d, Transform3d  # type: ignore[import-not-found]

from pipeline.coordinate_systems import openCvPoseToWpilib
from type_defs import FloatArray
from vision_types import FiducialPoseObservation


def inches_to_meters(inches: float) -> float:
    return inches * 0.0254


def meters_to_inches(meters: float) -> float:
    return meters / 0.0254


def _best_pose_vectors(
    observation: FiducialPoseObservation,
) -> tuple[FloatArray, FloatArray]:
    if (
        observation.error_1 is None
        or observation.tvec_1 is None
        or observation.rvec_1 is None
        or observation.error_0 <= observation.error_1
    ):
        return observation.tvec_0, observation.rvec_0
    return observation.tvec_1, observation.rvec_1


IMAGE_NAME = "manual_image_2.jpg"
REFERENCE_ID = 3
REFERENCE_POSE = Pose3d(
    inches_to_meters(610.1778104250433),
    inches_to_meters(171.52001866549656),
    inches_to_meters(17.163224225576677),
    Rotation3d(Quaternion(w=0.019125, x=-0.055698, y=0.016995, z=0.998120)),
)


if __name__ == "__main__":
    config = ConfigStore(LocalConfig(), RemoteConfig())
    local_config_source: ConfigSource = FileConfigSource()
    local_config_source.update(config)
    config.remote_config.fiducial_size_m = inches_to_meters(6.0)

    fiducial_detector = ArucoFiducialDetector(cv2.aruco.DICT_APRILTAG_16h5)
    pose_estimator = SquareTargetPoseEstimator()

    image = cv2.imread(IMAGE_NAME)
    if image is None:
        raise FileNotFoundError(IMAGE_NAME)
    image_observations = fiducial_detector.detect_fiducials(image, config)
    pose_observations = [
        observation
        for image_observation in image_observations
        for observation in [
            pose_estimator.solve_fiducial_pose(image_observation, config)
        ]
        if observation is not None
    ]

    camera_to_reference_pose = Pose3d()
    for observation in pose_observations:
        if observation.tag_id == REFERENCE_ID:
            best_tvec, best_rvec = _best_pose_vectors(observation)
            camera_to_reference_pose = openCvPoseToWpilib(best_tvec, best_rvec)
            break
    camera_to_reference = Transform3d(
        camera_to_reference_pose.translation(), camera_to_reference_pose.rotation()
    )
    field_to_camera_pose = REFERENCE_POSE.transformBy(camera_to_reference.inverse())

    for observation in pose_observations:
        best_tvec, best_rvec = _best_pose_vectors(observation)
        camera_to_tag_pose = openCvPoseToWpilib(best_tvec, best_rvec)
        camera_to_tag = Transform3d(
            camera_to_tag_pose.translation(), camera_to_tag_pose.rotation()
        )
        field_to_tag_pose = field_to_camera_pose.transformBy(camera_to_tag)
        print("Tag =", observation.tag_id)
        print("X =", meters_to_inches(field_to_tag_pose.X()), "inches")
        print("Y =", meters_to_inches(field_to_tag_pose.Y()), "inches")
        print("Z =", meters_to_inches(field_to_tag_pose.Z()), "inches")
        print("Rotation =", field_to_tag_pose.rotation().getQuaternion())
        print()
