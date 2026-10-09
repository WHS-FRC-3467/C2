import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import cv2
import numpy
import ntcore
from wpimath.geometry import Pose3d

from input.Config import ConfigStore, LocalConfig, RemoteConfig
from input.ConfigSource import ConfigSource, FileConfigSource, NTConfigSource
from input.RobotPoseSource import NTFeedbackSource, TimestampedPose
from output.OutputPublisher import NTFlatbufferOutputPublisher
from output.OverlayUtil import *
from output.CalibrationUploadServer import CalibrationUploadServer
from output.StreamServer import MjpegServer, RawCameraMjpegServer
from pipeline.CameraPoseEstimator import MultiTargetCameraPoseEstimator
from pipeline.Capture import MultiCameraCapture
from pipeline.ArucoNanoDetector import ArucoNanoFiducialDetector


def _run_aruco(config, remote_config_source):
    num_cameras = config.local_config.num_cameras

    capture = MultiCameraCapture()
    fiducial_detectors = [
        ArucoNanoFiducialDetector(cv2.aruco.DICT_APRILTAG_36h11)
        for _ in range(num_cameras)
    ]
    camera_pose_estimator = MultiTargetCameraPoseEstimator()
    output_publisher = NTFlatbufferOutputPublisher()
    robot_pose_source = NTFeedbackSource(config)
    stream_server = MjpegServer()
    raw_camera_servers = [RawCameraMjpegServer(i) for i in range(num_cameras)]
    detection_pool = ThreadPoolExecutor(max_workers=num_cameras)

    stream_server.start(config)
    base_port = config.local_config.stream_port
    for i, raw_server in enumerate(raw_camera_servers):
        port = base_port + 1 + i
        raw_server.start(port)
        print(f"Raw camera {i} stream at port {port}")

    frame_count = 0
    last_print = 0
    was_calibrating = False
    # Carry the last robot estimate forward as one starting guess for the next
    # frame. The current frame's detections still determine the new result.
    previous_vision_robot_pose = None
    while True:
        remote_config_source.update(config)
        t_cap0 = time.perf_counter()
        timestamped_image = capture.get_frame(config)
        t_cap1 = time.perf_counter()
        if timestamped_image is None:
            time.sleep(0.5)
            continue

        image, timestamp = timestamped_image

        fps = None
        frame_count += 1
        if time.time() - last_print > 1:
            last_print = time.time()
            fps = frame_count
            print("Running at", frame_count, "fps")
            frame_count = 0

        t0 = time.perf_counter()

        sub_frames = [
            numpy.ascontiguousarray(s)
            for s in MultiCameraCapture.split_frame(image, num_cameras)
        ]
        t_split = time.perf_counter()

        timestamped_seed_pose: Optional[TimestampedPose] = (
            robot_pose_source.get_pose()
        )
        if previous_vision_robot_pose is not None and (
            timestamped_seed_pose is None
            or previous_vision_robot_pose.timestamp_us
            > timestamped_seed_pose.timestamp_us
        ):
            timestamped_seed_pose = previous_vision_robot_pose

        seed_pose = (
            timestamped_seed_pose.pose
            if timestamped_seed_pose is not None
            else None
        )

        projected_tags_by_camera = (
            camera_pose_estimator.project_tag_corners_by_camera(
                seed_pose,
                config,
                [(frame.shape[1], frame.shape[0]) for frame in sub_frames],
            )
        )

        detect_futures = [
            detection_pool.submit(
                fiducial_detectors[i].detect_fiducials,
                sub_frames[i],
                projected_tags_by_camera[i],
            )
            for i in range(num_cameras)
        ]

        if len(image.shape) == 2:
            display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            display_image = image.copy()
        t_cvt = time.perf_counter()

        sub_width = image.shape[1] // num_cameras

        all_detections = [f.result() for f in detect_futures]
        t_det = time.perf_counter()

        for cam_idx in range(num_cameras):
            raw_camera_servers[cam_idx].set_frame(sub_frames[cam_idx])

            image_observations = all_detections[cam_idx]
            display_sub = display_image[
                :, cam_idx * sub_width : (cam_idx + 1) * sub_width
            ]
            overlay_crop_boxes(
                display_sub, fiducial_detectors[cam_idx].last_crop_boxes
            )
            [overlay_image_observation(display_sub, x) for x in image_observations]

        # Estimate one robot pose from the corners in all synchronized
        # subframes, rather than solving and publishing one pose per camera.
        robot_pose_observation = camera_pose_estimator.solve_robot_pose(
            all_detections,
            config,
            seed_pose,
        )
        if robot_pose_observation is not None:
            previous_vision_robot_pose = TimestampedPose(
                robot_pose_observation.pose_0, timestamp
            )

        output_publisher.send(
            config,
            timestamp,
            robot_pose_observation,
            fps,
        )
        t_pose = time.perf_counter()

        stream_server.set_frame(display_image)
        t_stream = time.perf_counter()

        if fps is not None:
            if camera_pose_estimator.last_failure_reason is not None:
                print("Pose unavailable:", camera_pose_estimator.last_failure_reason)
            elif robot_pose_observation is not None:
                print("Pose reprojection RMS (pixels):", ", ".join(
                    f"camera {index}: {error:.2f}"
                    for index, error in camera_pose_estimator.last_camera_errors.items()
                ))
            print(
                f"  cap:{(t_cap1-t_cap0)*1000:.1f} split:{(t_split-t0)*1000:.1f} cvt:{(t_cvt-t_split)*1000:.1f} "
                f"det:{(t_det-t_split)*1000:.1f} pose:{(t_pose-t_det)*1000:.1f} "
                f"stream:{(t_stream-t_pose)*1000:.1f} total:{(t_stream-t_cap0)*1000:.1f}ms"
            )
        continue


if __name__ == "__main__":
    config = ConfigStore(LocalConfig(), RemoteConfig())
    local_config_source: ConfigSource = FileConfigSource()
    remote_config_source: ConfigSource = NTConfigSource()

    local_config_source.update(config)

    ntcore.NetworkTableInstance.getDefault().setServer(config.local_config.server_ip)
    ntcore.NetworkTableInstance.getDefault().startClient4(config.local_config.device_id)

    calibration_upload_server = CalibrationUploadServer(
        port=config.local_config.stream_port + 10
    )
    calibration_upload_server.start()

    print("Starting in aruco detection mode")
    _run_aruco(config, remote_config_source)
