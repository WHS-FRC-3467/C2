from typing import List, Optional, Union

import cv2
import numpy
from scipy.optimize import least_squares
from wpimath.geometry import Pose3d, Quaternion, Rotation3d, Transform3d, Translation3d
from input.Config import ConfigStore
from pipeline.VisionTypes import CameraPoseObservation, FiducialImageObservation

from pipeline.CoordinateSystems import openCvPoseToWpilib, wpilibTranslationToOpenCv


_WPILIB_TO_OPENCV = numpy.array(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]],
    dtype=numpy.float64,
)
# WPILib uses +X forward, +Y left, +Z up. OpenCV camera coordinates use
# +X right, +Y down, +Z forward. This matrix converts a vector between them.


def _rotation_matrix(rotation: Rotation3d) -> numpy.ndarray:
    """Convert WPILib's quaternion rotation to a 3x3 matrix."""
    q = rotation.getQuaternion()
    w, x, y, z = q.W(), q.X(), q.Y(), q.Z()
    return numpy.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=numpy.float64,
    )


def _pose_matrix(pose: Pose3d) -> numpy.ndarray:
    """Return a homogeneous matrix for a pose: rotation in [:3,:3], position in [:3,3]."""
    matrix = numpy.eye(4, dtype=numpy.float64)
    matrix[:3, :3] = _rotation_matrix(pose.rotation())
    matrix[:3, 3] = [pose.X(), pose.Y(), pose.Z()]
    return matrix


def _pose_from_matrix(matrix: numpy.ndarray) -> Pose3d:
    """Convert the optimizer's rotation/translation matrix back to WPILib types."""
    rotation_vector, _ = cv2.Rodrigues(matrix[:3, :3])
    angle = float(numpy.linalg.norm(rotation_vector))
    if angle < 1e-12:
        rotation = Rotation3d()
    else:
        axis = rotation_vector.reshape(3) / angle
        rotation = Rotation3d(axis, angle)
    return Pose3d(
        Translation3d(*matrix[:3, 3].tolist()),
        rotation,
    )

class MultiTargetCameraPoseEstimator:
    last_failure_reason: Optional[str] = None
    MIN_PROJECTED_TAG_EDGE_PX = 10.0

    def __init__(self) -> None:
        self.last_camera_errors: dict[int, float] = {}
        self._field_tag_cache_key = None
        self._field_tag_corners: dict[int, numpy.ndarray] = {}

    def _get_field_tag_corners(self, remote) -> dict[int, numpy.ndarray]:
        # Compare values, since the NT source parses a new layout dict each frame.
        tags = []
        for tag in remote.tag_layout.get("tags", []):
            t = tag["pose"]["translation"]
            q = tag["pose"]["rotation"]["quaternion"]
            tags.append((tag["ID"], t["x"], t["y"], t["z"],
                         q["W"], q["X"], q["Y"], q["Z"]))
        key = (remote.fiducial_size_m, tuple(sorted(tags)))
        if key != self._field_tag_cache_key:
            half = remote.fiducial_size_m / 2.0
            offsets = numpy.asarray(
                [(0, half, -half), (0, -half, -half),
                 (0, -half, half), (0, half, half)], dtype=numpy.float64
            )
            corners = {}
            for tag_id, x, y, z, w, qx, qy, qz in tags:
                pose = Pose3d(Translation3d(x, y, z),
                              Rotation3d(Quaternion(w, qx, qy, qz)))
                matrix = _pose_matrix(pose)
                corners[tag_id] = offsets @ matrix[:3, :3].T + matrix[:3, 3]
            self._field_tag_corners = corners
            self._field_tag_cache_key = key
        return self._field_tag_corners

    def _pose_failed(self, reason: str) -> None:
        self.last_failure_reason = reason
        return None

    def project_tag_corners_by_camera(
        self, robot_pose: Optional[Pose3d], config_store: ConfigStore,
        image_sizes: Optional[List[tuple[int, int]]] = None,
    ) -> List[dict[int, numpy.ndarray]]:
        """Project the four nearest in-view field tags for each camera."""
        local = config_store.local_config
        remote = config_store.remote_config
        camera_count = local.num_cameras
        projected_by_camera: List[dict[int, numpy.ndarray]] = [
            {} for _ in range(camera_count)
        ]
        if (
            robot_pose is None
            or not remote.tag_layout
            or remote.fiducial_size_m <= 0
        ):
            return projected_by_camera

        field_from_robot = _pose_matrix(robot_pose)
        field_tags = self._get_field_tag_corners(remote)
        for camera_index in range(
            min(
                camera_count,
                len(remote.camera_extrinsics),
                len(local.camera_matrices),
                len(local.distortion_coefficients),
            )
        ):
            camera_matrix = numpy.asarray(local.camera_matrices[camera_index])
            distortion = numpy.asarray(local.distortion_coefficients[camera_index])
            if camera_matrix.size == 0 or distortion.size == 0:
                continue

            robot_from_camera = _pose_matrix(remote.camera_extrinsics[camera_index])
            field_from_camera_rotation = (
                field_from_robot[:3, :3] @ robot_from_camera[:3, :3]
            )
            field_from_camera_translation = (
                field_from_robot[:3, 3]
                + field_from_robot[:3, :3] @ robot_from_camera[:3, 3]
            )

            candidates = []
            width, height = (
                image_sizes[camera_index] if image_sizes is not None
                else (local.camera_resolution_width, local.camera_resolution_height)
            )
            for tag_id, field_points in field_tags.items():
                # Row-vector form of transforming field points into the
                # camera, then converting WPILib axes to OpenCV camera axes.
                camera_points_wp = (
                    field_points - field_from_camera_translation
                ) @ field_from_camera_rotation
                camera_points_cv = camera_points_wp @ _WPILIB_TO_OPENCV.T
                if numpy.any(camera_points_cv[:, 2] <= 1e-6):
                    continue
                pixels, _ = cv2.projectPoints(
                    camera_points_cv.reshape(-1, 1, 3),
                    numpy.zeros((3, 1)),
                    numpy.zeros((3, 1)),
                    camera_matrix,
                    distortion,
                )
                corners = pixels.reshape(4, 2)
                if not numpy.isfinite(corners).all():
                    continue
                edge_lengths = numpy.linalg.norm(
                    numpy.roll(corners, -1, axis=0) - corners, axis=1
                )
                if edge_lengths.min() < self.MIN_PROJECTED_TAG_EDGE_PX:
                    continue
                if width > 0 and height > 0:
                    low = corners.min(axis=0)
                    high = corners.max(axis=0)
                    if high[0] < 0 or high[1] < 0 or low[0] >= width or low[1] >= height:
                        continue
                distance = float(numpy.linalg.norm(
                    field_points.mean(axis=0) - field_from_camera_translation
                ))
                candidates.append((distance, tag_id, corners))
            candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
            projected_by_camera[camera_index] = {
                tag_id: corners for _, tag_id, corners in candidates[:4]
            }

        return projected_by_camera

    def solve_robot_pose(
        self,
        observations_by_camera: List[List[FiducialImageObservation]],
        config_store: ConfigStore,
        initial_pose: Optional[Pose3d] = None,
        yaw_prior: Optional[float] = None,
    ) -> Union[CameraPoseObservation, None]:
        """Estimate a ground-plane robot pose (X, Y, yaw) from all camera corners.

        Robot Z, roll, and pitch are fixed to zero. yaw_prior is an optional
        independent gyro-derived field yaw in radians, distinct from the seed.
        Its Gaussian uncertainty and corner noise scale come from local config.

        `observations_by_camera[i]` contains detections from camera i. Its
        intrinsic calibration and `camera_extrinsics[i]` must use that same
        index. Each extrinsic is the camera's fixed pose in robot coordinates.

        Transform names follow `field_from_robot`: that matrix takes a point
        expressed in robot coordinates and gives the same point in field
        coordinates. Thus `field_from_camera` is robot pose composed with the
        camera's fixed pose in the robot.

        The optimizer adjusts one candidate robot pose. For every camera, it
        composes that pose with the camera's fixed extrinsic, projects known
        field-space tag corners into that camera, and compares those pixels
        with the detector's measured corners. It minimizes all cameras' pixel
        errors together.
        """
        remote = config_store.remote_config
        local = config_store.local_config
        self.last_failure_reason = None
        self.last_camera_errors = {}
        if not remote.tag_layout or remote.fiducial_size_m <= 0:
            return self._pose_failed(
                f"Missing tag layout or invalid fiducial_size_m={remote.fiducial_size_m}"
            )
        if not observations_by_camera:
            return self._pose_failed("No camera observations supplied")
        if len(remote.camera_extrinsics) < len(observations_by_camera):
            return self._pose_failed(
                f"Need {len(observations_by_camera)} camera extrinsics, "
                f"received {len(remote.camera_extrinsics)}"
            )

        field_tags = self._get_field_tag_corners(remote)

        # Build one bundle per camera. Each bundle keeps field points paired
        # with image pixels and with that camera's own calibration/extrinsic.
        camera_data = []
        used_tag_ids = []
        for camera_index, camera_observations in enumerate(observations_by_camera):
            if not camera_observations:
                continue
            if camera_index >= len(local.camera_matrices) or camera_index >= len(
                local.distortion_coefficients
            ):
                return self._pose_failed(f"Missing calibration for camera {camera_index}")
            camera_matrix = local.camera_matrices[camera_index]
            distortion = local.distortion_coefficients[camera_index]
            if numpy.asarray(camera_matrix).size == 0 or numpy.asarray(distortion).size == 0:
                return self._pose_failed(f"Empty calibration for camera {camera_index}")

            field_points = []
            image_points = []
            for observation in camera_observations:
                tag_corners = field_tags.get(observation.tag_id)
                if tag_corners is None:
                    continue
                for corner_index, corner in enumerate(tag_corners):
                    field_points.append(corner.tolist())
                    # Detector output is commonly shaped (1, 4, 2); flattening
                    # to (4, 2) makes the four measured pixel pairs explicit.
                    used_corner = numpy.asarray(observation.corners).reshape(4, 2)[
                        corner_index
                    ]
                    image_points.append(used_corner.tolist())
                if observation.tag_id not in used_tag_ids:
                    used_tag_ids.append(observation.tag_id)

            if not field_points:
                continue

            camera_data.append(
                {
                    "index": camera_index,
                    "field_points": numpy.asarray(field_points, dtype=numpy.float64),
                    "image_points": numpy.asarray(image_points, dtype=numpy.float64),
                    "camera_matrix": numpy.asarray(camera_matrix, dtype=numpy.float64),
                    "distortion": numpy.asarray(distortion, dtype=numpy.float64),
                    "robot_to_camera": _pose_matrix(
                        remote.camera_extrinsics[camera_index]
                    ),
                }
            )

        if not camera_data or not used_tag_ids:
            detected_ids = sorted({
                observation.tag_id
                for observations in observations_by_camera
                for observation in observations
            })
            return self._pose_failed(
                f"No usable layout tags; detected IDs={detected_ids}"
            )

        # A nonlinear solver needs a starting pose. The previous frame is a
        # useful seed when available; independent per-camera PnP estimates add
        # fresh seeds. These PnP calls only initialize the search. The final
        # result is selected by the joint all-camera reprojection error below.
        seeds = []
        if initial_pose is not None:
            seeds.append(numpy.array([initial_pose.X(), initial_pose.Y(),
                                      initial_pose.rotation().Z()]))

        for camera in camera_data:
            field_points_cv = camera["field_points"] @ _WPILIB_TO_OPENCV.T
            try:
                solved, rvecs, tvecs, _ = cv2.solvePnPGeneric(
                    field_points_cv,
                    camera["image_points"],
                    camera["camera_matrix"],
                    camera["distortion"],
                    flags=cv2.SOLVEPNP_SQPNP,
                )
            except cv2.error:
                continue
            if not solved:
                continue

            robot_to_camera = camera["robot_to_camera"]
            for rvec, tvec in zip(rvecs, tvecs):
                # PnP returns the transform from field/object coordinates to
                # this camera's OpenCV coordinates. Convert it to the camera's
                # pose in the field, then remove the fixed camera-in-robot
                # offset to get a candidate robot pose in the field.
                camera_from_field_cv, _ = cv2.Rodrigues(rvec)
                camera_from_field_wp = (
                    _WPILIB_TO_OPENCV.T
                    @ camera_from_field_cv
                    @ _WPILIB_TO_OPENCV
                )
                field_from_camera_rotation = camera_from_field_wp.T
                field_from_camera_translation = _WPILIB_TO_OPENCV.T @ (
                    -camera_from_field_cv.T @ numpy.asarray(tvec).reshape(3)
                )
                field_from_robot_rotation = (
                    field_from_camera_rotation @ robot_to_camera[:3, :3].T
                )
                field_from_robot_translation = field_from_camera_translation - (
                    field_from_robot_rotation @ robot_to_camera[:3, 3]
                )
                seeds.append(numpy.array([
                    field_from_robot_translation[0], field_from_robot_translation[1],
                    numpy.arctan2(field_from_robot_rotation[1, 0],
                                  field_from_robot_rotation[0, 0]),
                ]))

        if not seeds:
            return self._pose_failed("PnP produced no pose seeds and no initial pose supplied")

        prior_weight = 0.0
        if yaw_prior is not None:
            sigma_yaw = numpy.deg2rad(local.yaw_prior_stddev_deg)
            sigma_pixel = local.corner_noise_stddev_px
            if (not numpy.isfinite(yaw_prior) or not numpy.isfinite(sigma_yaw)
                    or sigma_yaw <= 0 or not numpy.isfinite(sigma_pixel)
                    or sigma_pixel <= 0):
                return self._pose_failed("Invalid yaw prior or uncertainty settings")
            prior_weight = sigma_pixel / sigma_yaw

        def pixel_residual(parameters):
            """Return predicted-minus-observed pixel errors for every corner.

            Parameters are [robot field X, Y, yaw radians].
            """
            c, s = numpy.cos(parameters[2]), numpy.sin(parameters[2])
            field_from_robot_rotation = numpy.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            field_from_robot_translation = numpy.array([parameters[0], parameters[1], 0.0])
            all_residuals = []
            for camera in camera_data:
                robot_to_camera = camera["robot_to_camera"]
                # Compose the robot's field pose with this camera's fixed
                # robot-relative pose. This gives the camera's position and
                # orientation in the field. To find where a field point lies
                # in the camera, subtract the camera position and undo rotation.
                field_from_camera_rotation = (
                    field_from_robot_rotation @ robot_to_camera[:3, :3]
                )
                field_from_camera_translation = (
                    field_from_robot_translation
                    + field_from_robot_rotation @ robot_to_camera[:3, 3]
                )
                # Points are rows in this NumPy array. Multiplication by the
                # rotation matrix here is the row-vector form of undoing the
                # camera rotation on each (point - camera_position) vector.
                camera_points_wp = (
                    camera["field_points"] - field_from_camera_translation
                ) @ field_from_camera_rotation
                # Convert camera-frame points to OpenCV axes, then ask OpenCV
                # to apply this camera's K matrix and lens-distortion model.
                camera_points_cv = camera_points_wp @ _WPILIB_TO_OPENCV.T
                depths = camera_points_cv[:, 2]
                projected, _ = cv2.projectPoints(
                    camera_points_cv.reshape(-1, 1, 3),
                    numpy.zeros((3, 1)),
                    numpy.zeros((3, 1)),
                    camera["camera_matrix"],
                    camera["distortion"],
                )
                pixel_residuals = (
                    projected.reshape(-1, 2) - camera["image_points"]
                )
                # A point behind the lens cannot produce a valid observation.
                # Give it a large penalty so the optimizer moves it in front.
                behind = depths <= 1e-6
                pixel_residuals[behind] = (
                    1000.0 + numpy.abs(depths[behind, numpy.newaxis]) * 100.0
                )
                all_residuals.append(pixel_residuals.reshape(-1))
            return numpy.concatenate(all_residuals)

        def residual(parameters):
            pixels = pixel_residual(parameters)
            if yaw_prior is None:
                return pixels
            # Shortest angular difference, including the +/-pi boundary.
            delta = numpy.arctan2(numpy.sin(parameters[2] - yaw_prior),
                                 numpy.cos(parameters[2] - yaw_prior))
            return numpy.append(pixels, delta * prior_weight)

        best_parameters = None
        best_cost = float("inf")
        solver_failures = []
        for seed in seeds:
            try:
                # LM adjusts the three planar pose numbers to reduce the pixel residuals.
                # `2-point` estimates the Jacobian by slightly perturbing each
                # parameter; `x_scale="jac"` balances meters and radians.
                result = least_squares(
                    residual,
                    seed,
                    method="lm",
                    jac="2-point",
                    x_scale="jac",
                    max_nfev=200,
                )
            except (ValueError, cv2.error, numpy.linalg.LinAlgError) as error:
                solver_failures.append(str(error))
                continue
            # Compare seeds using image error plus the optional yaw prior. Keep only a
            # converged finite solution with the lowest joint reprojection cost.
            cost = float(result.fun @ result.fun)
            if result.success and numpy.isfinite(cost) and cost < best_cost:
                best_parameters, best_cost = result.x, cost
            elif not result.success or not numpy.isfinite(cost):
                solver_failures.append(f"{result.message}; cost={cost}")

        if best_parameters is None or not numpy.isfinite(best_cost):
            return self._pose_failed(
                "No converged finite pose solution: " + "; ".join(solver_failures)
            )
        robot_pose = Pose3d(Translation3d(best_parameters[0], best_parameters[1], 0.0),
                            Rotation3d(0.0, 0.0, best_parameters[2]))
        point_count = sum(camera["image_points"].shape[0] for camera in camera_data)
        # RMS is the typical 2D corner error in pixels: sum(dx²+dy²) divided
        # by the number of corners, then square-rooted.
        # Report only image error; the yaw prior is not a measured corner.
        final_residuals = pixel_residual(best_parameters)
        rms_error = float(numpy.sqrt(final_residuals @ final_residuals / point_count))
        offset = 0
        for camera in camera_data:
            count = camera["image_points"].shape[0]
            camera_residuals = final_residuals[offset:offset + 2 * count]
            self.last_camera_errors[camera["index"]] = float(
                numpy.sqrt(camera_residuals @ camera_residuals / count)
            )
            offset += 2 * count
        return CameraPoseObservation(
            used_tag_ids, robot_pose, rms_error, None, None
        )

    def solve_camera_pose(
        self,
        camera_index: int,
        image_observations: List[FiducialImageObservation],
        config_store: ConfigStore,
    ) -> Union[CameraPoseObservation, None]:
        """Estimate a camera pose from one camera's detections independently.

        The main capture loop uses solve_robot_pose() for the combined rig
        estimate. This method remains available for callers that need the
        older, single-camera result.
        """
        # Exit if no tag layout available
        if config_store.remote_config.tag_layout == None:
            return None

        # Exit if no observations available
        if len(image_observations) == 0:
            return None

        # Create set of object and image points
        fid_size = config_store.remote_config.fiducial_size_m
        object_points = []
        image_points = []
        tag_ids = []
        tag_poses = []
        for observation in image_observations:
            tag_pose = None
            for tag_data in config_store.remote_config.tag_layout["tags"]:
                if tag_data["ID"] == observation.tag_id:
                    tag_pose = Pose3d(
                        Translation3d(
                            tag_data["pose"]["translation"]["x"],
                            tag_data["pose"]["translation"]["y"],
                            tag_data["pose"]["translation"]["z"],
                        ),
                        Rotation3d(
                            Quaternion(
                                tag_data["pose"]["rotation"]["quaternion"]["W"],
                                tag_data["pose"]["rotation"]["quaternion"]["X"],
                                tag_data["pose"]["rotation"]["quaternion"]["Y"],
                                tag_data["pose"]["rotation"]["quaternion"]["Z"],
                            )
                        ),
                    )
            if tag_pose != None:
                # Add object points by transforming from the tag center
                corner_0 = tag_pose + Transform3d(
                    Translation3d(0, fid_size / 2.0, -fid_size / 2.0), Rotation3d()
                )
                corner_1 = tag_pose + Transform3d(
                    Translation3d(0, -fid_size / 2.0, -fid_size / 2.0), Rotation3d()
                )
                corner_2 = tag_pose + Transform3d(
                    Translation3d(0, -fid_size / 2.0, fid_size / 2.0), Rotation3d()
                )
                corner_3 = tag_pose + Transform3d(
                    Translation3d(0, fid_size / 2.0, fid_size / 2.0), Rotation3d()
                )
                object_points += [
                    wpilibTranslationToOpenCv(corner_0.translation()),
                    wpilibTranslationToOpenCv(corner_1.translation()),
                    wpilibTranslationToOpenCv(corner_2.translation()),
                    wpilibTranslationToOpenCv(corner_3.translation()),
                ]

                # Add image points from observation
                image_points += [
                    [observation.corners[0][0][0], observation.corners[0][0][1]],
                    [observation.corners[0][1][0], observation.corners[0][1][1]],
                    [observation.corners[0][2][0], observation.corners[0][2][1]],
                    [observation.corners[0][3][0], observation.corners[0][3][1]],
                ]

                # Add tag ID and pose
                tag_ids.append(observation.tag_id)
                tag_poses.append(tag_pose)

        # Single tag, return two poses
        if len(tag_ids) == 1:
            object_points = numpy.array(
                [
                    [-fid_size / 2.0, fid_size / 2.0, 0.0],
                    [fid_size / 2.0, fid_size / 2.0, 0.0],
                    [fid_size / 2.0, -fid_size / 2.0, 0.0],
                    [-fid_size / 2.0, -fid_size / 2.0, 0.0],
                ]
            )
            try:
                _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                    object_points,
                    numpy.array(image_points),
                    config_store.local_config.camera_matrices[camera_index],
                    config_store.local_config.distortion_coefficients[camera_index],
                    flags=cv2.SOLVEPNP_IPPE_SQUARE,
                )
            except:
                return None

            # Calculate WPILib camera poses
            field_to_tag_pose = tag_poses[0]
            camera_to_tag_pose_0 = openCvPoseToWpilib(tvecs[0], rvecs[0])
            camera_to_tag_pose_1 = openCvPoseToWpilib(tvecs[1], rvecs[1])
            camera_to_tag_0 = Transform3d(
                camera_to_tag_pose_0.translation(), camera_to_tag_pose_0.rotation()
            )
            camera_to_tag_1 = Transform3d(
                camera_to_tag_pose_1.translation(), camera_to_tag_pose_1.rotation()
            )
            field_to_camera_0 = field_to_tag_pose.transformBy(camera_to_tag_0.inverse())
            field_to_camera_1 = field_to_tag_pose.transformBy(camera_to_tag_1.inverse())
            field_to_camera_pose_0 = Pose3d(
                field_to_camera_0.translation(), field_to_camera_0.rotation()
            )
            field_to_camera_pose_1 = Pose3d(
                field_to_camera_1.translation(), field_to_camera_1.rotation()
            )

            # Return result
            return CameraPoseObservation(
                tag_ids,
                field_to_camera_pose_0,
                errors[0][0],
                field_to_camera_pose_1,
                errors[1][0],
            )

        # Multi-tag, return one pose
        else:
            # Run SolvePNP with all tags
            try:
                _, rvecs, tvecs, errors = cv2.solvePnPGeneric(
                    numpy.array(object_points),
                    numpy.array(image_points),
                    config_store.local_config.camera_matrices[camera_index],
                    config_store.local_config.distortion_coefficients[camera_index],
                    flags=cv2.SOLVEPNP_SQPNP,
                )
            except:
                return None

            # Calculate WPILib camera pose
            camera_to_field_pose = openCvPoseToWpilib(tvecs[0], rvecs[0])
            camera_to_field = Transform3d(
                camera_to_field_pose.translation(), camera_to_field_pose.rotation()
            )
            field_to_camera = camera_to_field.inverse()
            field_to_camera_pose = Pose3d(
                field_to_camera.translation(), field_to_camera.rotation()
            )

            # Return result
            return CameraPoseObservation(
                tag_ids, field_to_camera_pose, errors[0][0], None, None
            )
