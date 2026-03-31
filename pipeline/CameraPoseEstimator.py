from typing import List, Union

import cv2
import numpy
from scipy.optimize import least_squares
from config.config import ConfigStore
from vision_types import CameraPoseObservation, FiducialImageObservation
from wpimath.geometry import *

from pipeline.coordinate_systems import (openCvPoseToWpilib,
                                         wpilibPoseToOpenCv,
                                         wpilibTranslationToOpenCv)


class CameraPoseEstimator:
    def __init__(self) -> None:
        raise NotImplementedError

    def solve_camera_pose(self, image_observations: List[FiducialImageObservation], config_store: ConfigStore) -> Union[CameraPoseObservation, None]:
        raise NotImplementedError


class MultiTargetCameraPoseEstimator(CameraPoseEstimator):
    def __init__(self) -> None:
        pass

    def solve_camera_pose(self, image_observations: List[FiducialImageObservation], config_store: ConfigStore) -> Union[CameraPoseObservation, None]:
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
                            tag_data["pose"]["translation"]["z"]
                        ),
                        Rotation3d(Quaternion(
                            tag_data["pose"]["rotation"]["quaternion"]["W"],
                            tag_data["pose"]["rotation"]["quaternion"]["X"],
                            tag_data["pose"]["rotation"]["quaternion"]["Y"],
                            tag_data["pose"]["rotation"]["quaternion"]["Z"]
                        )))
            if tag_pose != None:
                # Add object points by transforming from the tag center
                corner_0 = tag_pose + Transform3d(Translation3d(0, fid_size / 2.0, -fid_size / 2.0), Rotation3d())
                corner_1 = tag_pose + Transform3d(Translation3d(0, -fid_size / 2.0, -fid_size / 2.0), Rotation3d())
                corner_2 = tag_pose + Transform3d(Translation3d(0, -fid_size / 2.0, fid_size / 2.0), Rotation3d())
                corner_3 = tag_pose + Transform3d(Translation3d(0, fid_size / 2.0, fid_size / 2.0), Rotation3d())
                object_points += [
                    wpilibTranslationToOpenCv(corner_0.translation()),
                    wpilibTranslationToOpenCv(corner_1.translation()),
                    wpilibTranslationToOpenCv(corner_2.translation()),
                    wpilibTranslationToOpenCv(corner_3.translation())
                ]

                # Add image points from observation
                image_points += [
                    [observation.corners[0][0][0], observation.corners[0][0][1]],
                    [observation.corners[0][1][0], observation.corners[0][1][1]],
                    [observation.corners[0][2][0], observation.corners[0][2][1]],
                    [observation.corners[0][3][0], observation.corners[0][3][1]]
                ]

                # Add tag ID and pose
                tag_ids.append(observation.tag_id)
                tag_poses.append(tag_pose)

        # Single tag, return two poses
        if len(tag_ids) == 1:
            object_points = numpy.array([[-fid_size / 2.0, fid_size / 2.0, 0.0],
                                         [fid_size / 2.0, fid_size / 2.0, 0.0],
                                         [fid_size / 2.0, -fid_size / 2.0, 0.0],
                                         [-fid_size / 2.0, -fid_size / 2.0, 0.0]])
            try:
                _, rvecs, tvecs, errors = cv2.solvePnPGeneric(object_points, numpy.array(image_points),
                                                              config_store.local_config.camera_matrix, config_store.local_config.distortion_coefficients, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            except:
                return None

            # Calculate WPILib camera poses
            field_to_tag_pose = tag_poses[0]
            camera_to_tag_pose_0 = openCvPoseToWpilib(tvecs[0], rvecs[0])
            camera_to_tag_pose_1 = openCvPoseToWpilib(tvecs[1], rvecs[1])
            camera_to_tag_0 = Transform3d(camera_to_tag_pose_0.translation(), camera_to_tag_pose_0.rotation())
            camera_to_tag_1 = Transform3d(camera_to_tag_pose_1.translation(), camera_to_tag_pose_1.rotation())
            field_to_camera_0 = field_to_tag_pose.transformBy(camera_to_tag_0.inverse())
            field_to_camera_1 = field_to_tag_pose.transformBy(camera_to_tag_1.inverse())
            field_to_camera_pose_0 = Pose3d(field_to_camera_0.translation(), field_to_camera_0.rotation())
            field_to_camera_pose_1 = Pose3d(field_to_camera_1.translation(), field_to_camera_1.rotation())

            # Return result
            return CameraPoseObservation(tag_ids, field_to_camera_pose_0, errors[0][0], field_to_camera_pose_1, errors[1][0])

        # Multi-tag, return one pose
        else:
            # Run SolvePNP with all tags
            try:
                _, rvecs, tvecs, errors = cv2.solvePnPGeneric(numpy.array(object_points), numpy.array(image_points),
                                                              config_store.local_config.camera_matrix, config_store.local_config.distortion_coefficients, flags=cv2.SOLVEPNP_SQPNP)
            except:
                return None

            # Calculate WPILib camera pose
            camera_to_field_pose = openCvPoseToWpilib(tvecs[0], rvecs[0])
            camera_to_field = Transform3d(camera_to_field_pose.translation(), camera_to_field_pose.rotation())
            field_to_camera = camera_to_field.inverse()
            field_to_camera_pose = Pose3d(field_to_camera.translation(), field_to_camera.rotation())

            # Return result
            return CameraPoseObservation(tag_ids, field_to_camera_pose, errors[0][0], None, None)


class MultiCameraFusedPoseEstimator:
    """Solves for field_to_robot using observations from multiple cameras jointly.

    Uses Levenberg-Marquardt minimization of reprojection error across all cameras.
    Requires known camera intrinsics and robot_to_camera extrinsics for each camera.
    """

    def __init__(self) -> None:
        pass

    def solve_fused_pose(
        self,
        all_detections: List[List[FiducialImageObservation]],
        config_store: ConfigStore
    ) -> Union[CameraPoseObservation, None]:
        if config_store.remote_config.tag_layout is None:
            return None

        extrinsics = config_store.local_config.camera_extrinsics
        if len(extrinsics) == 0:
            return None

        fid_size = config_store.remote_config.fiducial_size_m

        # Build per-camera observation lists: (object_points_3d, image_points_2d, cam_idx)
        camera_obs = []  # list of (obj_pts Nx3, img_pts Nx2, cam_idx, tag_ids)
        all_tag_ids = set()
        best_cam_idx = -1
        best_cam_count = 0

        for cam_idx, detections in enumerate(all_detections):
            if cam_idx >= len(extrinsics):
                continue
            if not config_store.local_config.has_calibrations[cam_idx] if cam_idx < len(config_store.local_config.has_calibrations) else True:
                continue
            if len(detections) == 0:
                continue

            obj_pts = []
            img_pts = []
            cam_tag_ids = []

            for obs in detections:
                tag_pose = None
                for tag_data in config_store.remote_config.tag_layout["tags"]:
                    if tag_data["ID"] == obs.tag_id:
                        tag_pose = Pose3d(
                            Translation3d(
                                tag_data["pose"]["translation"]["x"],
                                tag_data["pose"]["translation"]["y"],
                                tag_data["pose"]["translation"]["z"]
                            ),
                            Rotation3d(Quaternion(
                                tag_data["pose"]["rotation"]["quaternion"]["W"],
                                tag_data["pose"]["rotation"]["quaternion"]["X"],
                                tag_data["pose"]["rotation"]["quaternion"]["Y"],
                                tag_data["pose"]["rotation"]["quaternion"]["Z"]
                            )))
                if tag_pose is None:
                    continue

                corner_0 = tag_pose + Transform3d(Translation3d(0, fid_size / 2.0, -fid_size / 2.0), Rotation3d())
                corner_1 = tag_pose + Transform3d(Translation3d(0, -fid_size / 2.0, -fid_size / 2.0), Rotation3d())
                corner_2 = tag_pose + Transform3d(Translation3d(0, -fid_size / 2.0, fid_size / 2.0), Rotation3d())
                corner_3 = tag_pose + Transform3d(Translation3d(0, fid_size / 2.0, fid_size / 2.0), Rotation3d())
                obj_pts += [
                    wpilibTranslationToOpenCv(corner_0.translation()),
                    wpilibTranslationToOpenCv(corner_1.translation()),
                    wpilibTranslationToOpenCv(corner_2.translation()),
                    wpilibTranslationToOpenCv(corner_3.translation())
                ]
                img_pts += [
                    [obs.corners[0][0][0], obs.corners[0][0][1]],
                    [obs.corners[0][1][0], obs.corners[0][1][1]],
                    [obs.corners[0][2][0], obs.corners[0][2][1]],
                    [obs.corners[0][3][0], obs.corners[0][3][1]]
                ]
                cam_tag_ids.append(obs.tag_id)
                all_tag_ids.add(obs.tag_id)

            if len(cam_tag_ids) > 0:
                camera_obs.append((numpy.array(obj_pts), numpy.array(img_pts), cam_idx, cam_tag_ids))
                if len(cam_tag_ids) > best_cam_count:
                    best_cam_count = len(cam_tag_ids)
                    best_cam_idx = cam_idx

        # Need observations from at least 2 cameras
        cameras_with_obs = len(camera_obs)
        if cameras_with_obs < 2:
            return None

        # Initial guess: solvePnP on camera with most detections, then compose with extrinsic
        seed_obj, seed_img, seed_cam_idx, _ = camera_obs[0]
        for entry in camera_obs:
            if entry[2] == best_cam_idx:
                seed_obj, seed_img, seed_cam_idx, _ = entry
                break

        cam_matrix = config_store.local_config.camera_matrices[seed_cam_idx]
        dist_coeffs = config_store.local_config.distortion_coefficients_list[seed_cam_idx]
        seed_tag_ids = camera_obs[[e[2] for e in camera_obs].index(seed_cam_idx)][3]

        try:
            if len(seed_tag_ids) == 1:
                # Single tag on seed camera — use IPPE_SQUARE with tag-relative points
                ippe_obj = numpy.array([[-fid_size / 2.0, fid_size / 2.0, 0.0],
                                        [fid_size / 2.0, fid_size / 2.0, 0.0],
                                        [fid_size / 2.0, -fid_size / 2.0, 0.0],
                                        [-fid_size / 2.0, -fid_size / 2.0, 0.0]])
                _, rvecs_ippe, tvecs_ippe, errors_ippe = cv2.solvePnPGeneric(
                    ippe_obj, seed_img, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                # Pick the solution with lower reprojection error
                best_idx = 0 if errors_ippe[0][0] <= errors_ippe[1][0] else 1
                rvec_init, tvec_init = rvecs_ippe[best_idx], tvecs_ippe[best_idx]
                # camera_to_tag from IPPE, compose with field_to_tag to get camera_to_field
                camera_to_tag_pose = openCvPoseToWpilib(tvec_init, rvec_init)
                camera_to_tag = Transform3d(camera_to_tag_pose.translation(), camera_to_tag_pose.rotation())
                # Find field_to_tag for the seed tag
                seed_tag_pose = None
                for tag_data in config_store.remote_config.tag_layout["tags"]:
                    if tag_data["ID"] == seed_tag_ids[0]:
                        seed_tag_pose = Pose3d(
                            Translation3d(
                                tag_data["pose"]["translation"]["x"],
                                tag_data["pose"]["translation"]["y"],
                                tag_data["pose"]["translation"]["z"]),
                            Rotation3d(Quaternion(
                                tag_data["pose"]["rotation"]["quaternion"]["W"],
                                tag_data["pose"]["rotation"]["quaternion"]["X"],
                                tag_data["pose"]["rotation"]["quaternion"]["Y"],
                                tag_data["pose"]["rotation"]["quaternion"]["Z"])))
                field_to_camera_pose = seed_tag_pose.transformBy(camera_to_tag.inverse())
                field_to_camera = Transform3d(field_to_camera_pose.translation(), field_to_camera_pose.rotation())
            else:
                # Multiple tags on seed camera — use SQPNP with field-frame points
                success, rvec_init, tvec_init = cv2.solvePnP(
                    seed_obj, seed_img, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)
                if not success:
                    return None
                camera_to_field_pose = openCvPoseToWpilib(tvec_init, rvec_init)
                camera_to_field = Transform3d(camera_to_field_pose.translation(), camera_to_field_pose.rotation())
                field_to_camera = camera_to_field.inverse()
        except:
            return None

        robot_to_camera = Transform3d(extrinsics[seed_cam_idx].translation(), extrinsics[seed_cam_idx].rotation())
        field_to_robot = Pose3d(field_to_camera.translation(), field_to_camera.rotation()).transformBy(robot_to_camera.inverse())

        # Convert initial field_to_robot to OpenCV parameterization for optimization
        rvec0, tvec0 = wpilibPoseToOpenCv(Pose3d(field_to_robot.translation(), field_to_robot.rotation()))
        x0 = numpy.array([rvec0[0, 0], rvec0[1, 0], rvec0[2, 0],
                           tvec0[0, 0], tvec0[1, 0], tvec0[2, 0]])

        # Precompute per-camera field_to_camera extrinsic in OpenCV frame
        cam_rvecs = []
        cam_tvecs = []
        cam_matrices = []
        cam_dist_coeffs = []
        cam_obj_pts = []
        cam_img_pts = []

        for obj_pts, img_pts, cam_idx, _ in camera_obs:
            robot_to_cam = extrinsics[cam_idx]
            rvec_ext, tvec_ext = wpilibPoseToOpenCv(robot_to_cam)
            cam_rvecs.append(rvec_ext)
            cam_tvecs.append(tvec_ext)
            cam_matrices.append(config_store.local_config.camera_matrices[cam_idx])
            cam_dist_coeffs.append(config_store.local_config.distortion_coefficients_list[cam_idx])
            cam_obj_pts.append(obj_pts)
            cam_img_pts.append(img_pts)

        def residuals(x):
            # x = [rx, ry, rz, tx, ty, tz] for field_to_robot in OpenCV frame
            rvec_robot = numpy.array([[x[0]], [x[1]], [x[2]]])
            tvec_robot = numpy.array([[x[3]], [x[4]], [x[5]]])
            R_robot, _ = cv2.Rodrigues(rvec_robot)

            all_residuals = []
            for i in range(len(cam_rvecs)):
                # Compose: field_to_camera = robot_to_camera * field_to_robot
                R_ext, _ = cv2.Rodrigues(cam_rvecs[i])
                R_cam = R_ext @ R_robot
                t_cam = R_ext @ tvec_robot + cam_tvecs[i]
                rvec_cam, _ = cv2.Rodrigues(R_cam)

                projected, _ = cv2.projectPoints(
                    cam_obj_pts[i], rvec_cam, t_cam,
                    cam_matrices[i], cam_dist_coeffs[i])
                projected = projected.reshape(-1, 2)
                diff = projected - cam_img_pts[i]
                all_residuals.append(diff.ravel())

            return numpy.concatenate(all_residuals)

        try:
            result = least_squares(residuals, x0, method='lm')
        except:
            return None

        # Convert optimized result back to WPILib field_to_robot
        opt_rvec = numpy.array([[result.x[0]], [result.x[1]], [result.x[2]]])
        opt_tvec = numpy.array([[result.x[3]], [result.x[4]], [result.x[5]]])
        field_to_robot_pose_cv = openCvPoseToWpilib(opt_tvec, opt_rvec)
        field_to_robot_pose = Pose3d(field_to_robot_pose_cv.translation(), field_to_robot_pose_cv.rotation())

        # Compute mean reprojection error
        final_residuals = residuals(result.x)
        num_points = len(final_residuals) // 2
        reproj_errors = final_residuals.reshape(-1, 2)
        mean_error = float(numpy.mean(numpy.linalg.norm(reproj_errors, axis=1)))

        return CameraPoseObservation(
            sorted(all_tag_ids), field_to_robot_pose, mean_error, None, None)
