import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import cv2
import numpy
import ntcore

from calibration.CalibrationCommandSource import (CalibrationCommandSource,
                                                  NTCalibrationCommandSource)
from calibration.CalibrationSession import CalibrationSession
from config.config import ConfigStore, LocalConfig, RemoteConfig
from config.ConfigSource import ConfigSource, FileConfigSource, NTConfigSource
from output.ObjectDetectionPublisher import NTObjectDetectionPublisher
from output.OutputPublisher import NTFlatbufferOutputPublisher
from output.overlay_util import *
from output.StreamServer import MjpegServer, RawCameraMjpegServer
from pipeline.CameraPoseEstimator import MultiTargetCameraPoseEstimator
from pipeline.Capture import MultiCameraCapture
from pipeline.ArucoNanoDetector import ArucoNanoFiducialDetector
from pipeline.TensorRTYoloDetector import TensorRTDetectorError, TensorRTYoloDetector
from pipeline.VideoDeviceCapture import VideoDeviceCapture

OBJDETECT_DEVICE = "/dev/video1"
OBJDETECT_CALIBRATION = "calibration_objdetect.yml"
YOLO_ENGINE = "./model.engine"
YOLO_ONNX = "./model.onnx"

def _load_camera_matrix(filename: str) -> numpy.ndarray:
    filenames = [filename]
    if filename.endswith(".yml"):
        filenames.append(filename[:-4] + ".json")
    elif filename.endswith(".json"):
        filenames.append(filename[:-5] + ".yml")

    for candidate in filenames:
        if not os.path.exists(candidate):
            continue
        calibration_store = cv2.FileStorage(candidate, cv2.FILE_STORAGE_READ)
        camera_matrix = calibration_store.getNode("camera_matrix").mat()
        calibration_store.release()
        if isinstance(camera_matrix, numpy.ndarray):
            return camera_matrix
    return numpy.array([])


def _copy_bgr_frame(frame: cv2.Mat) -> cv2.Mat:
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return frame.copy()


def _clear_aruco_outputs(output_publishers, config: ConfigStore, timestamp: float) -> None:
    for cam_idx, output_publisher in enumerate(output_publishers):
        cam_config = config.for_camera(cam_idx)
        output_publisher.send(cam_config, timestamp, None, 0 if cam_idx == 0 else None)

if __name__ == "__main__":
    config = ConfigStore(LocalConfig(), RemoteConfig())
    local_config_source: ConfigSource = FileConfigSource()
    remote_config_source: ConfigSource = NTConfigSource()
    calibration_command_source: CalibrationCommandSource = NTCalibrationCommandSource()

    local_config_source.update(config)
    num_cameras = config.local_config.num_cameras
    object_detection_mode = config.local_config.enable_object_detection

    camera_pose_estimator = MultiTargetCameraPoseEstimator()
    output_publishers = [NTFlatbufferOutputPublisher(i) for i in range(num_cameras)]
    object_detection_publisher = NTObjectDetectionPublisher(config.local_config.device_id)
    stream_server = MjpegServer()
    calibration_session = CalibrationSession()
    capture = None
    fiducial_detectors = []
    raw_camera_servers = []
    detection_pool = None
    if not object_detection_mode:
        capture = MultiCameraCapture()
        # Per-camera detector instances for thread safety (C library releases GIL)
        fiducial_detectors = [ArucoNanoFiducialDetector(cv2.aruco.DICT_APRILTAG_36h11)
                              for _ in range(num_cameras)]
        raw_camera_servers = [RawCameraMjpegServer(i) for i in range(num_cameras)]
        detection_pool = ThreadPoolExecutor(max_workers=num_cameras)

    video1_capture = None
    video1_detector = None
    video1_error = None
    if object_detection_mode:
        video1_capture = VideoDeviceCapture(OBJDETECT_DEVICE)
        video1_camera_matrix = _load_camera_matrix(OBJDETECT_CALIBRATION)
        if video1_camera_matrix.size != 0:
            video1_detector = TensorRTYoloDetector(YOLO_ENGINE, YOLO_ONNX, video1_camera_matrix)
        else:
            video1_error = f"Missing camera matrix in {OBJDETECT_CALIBRATION}"
            print(video1_error)

    ntcore.NetworkTableInstance.getDefault().setServer(config.local_config.server_ip)
    ntcore.NetworkTableInstance.getDefault().startClient4(config.local_config.device_id)
    stream_server.start(config)
    print(f"Detection stream at port {config.local_config.stream_port}")
    base_port = config.local_config.stream_port
    if object_detection_mode:
        print("Object detection mode enabled; ArUco detections disabled")
    else:
        print("ArUco detection mode enabled; object detection disabled")
        for i, raw_server in enumerate(raw_camera_servers):
            port = base_port + 1 + i
            raw_server.start(port)
            print(f"Raw camera {i} stream at port {port}")

    frame_count = 0
    last_print = 0
    was_calibrating = False
    while True:
        remote_config_source.update(config)
        timestamp = time.time()

        if object_detection_mode:
            _clear_aruco_outputs(output_publishers, config, timestamp)
            video1_success, video1_frame = video1_capture.read()
            if not video1_success:
                object_detection_publisher.send(timestamp, [])
                if video1_error != f"Unable to read {OBJDETECT_DEVICE}":
                    video1_error = f"Unable to read {OBJDETECT_DEVICE}"
                    print(video1_error)
                time.sleep(0.05)
                continue

            display_frame = _copy_bgr_frame(video1_frame)
            if video1_detector is not None:
                try:
                    video1_detections = video1_detector.detect(video1_frame)
                    object_detection_publisher.send(timestamp, video1_detections)
                    for detection in video1_detections:
                        overlay_object_detection(display_frame, detection)
                    video1_error = None
                except TensorRTDetectorError as exc:
                    if video1_error != str(exc):
                        video1_error = str(exc)
                        print(f"video1 YOLO disabled: {video1_error}")
                    object_detection_publisher.send(timestamp, [])
            else:
                object_detection_publisher.send(timestamp, [])

            stream_server.set_frame(display_frame)
            continue

        object_detection_publisher.send(timestamp, [])

        t_cap0 = time.perf_counter()
        success, image = capture.get_frame(config)
        t_cap1 = time.perf_counter()
        if not success:
            time.sleep(0.5)
            continue

        fps = None
        frame_count += 1
        if time.time() - last_print > 1:
            last_print = time.time()
            fps = frame_count
            print("Running at", frame_count, "fps")
            frame_count = 0

        if calibration_command_source.get_calibrating(config):
            # Calibration mode
            was_calibrating = True
            calibration_session.process_frame(image, calibration_command_source.get_capture_flag(config))

        elif was_calibrating:
            # Finish calibration
            calibration_session.finish()
            sys.exit(0)

        elif config.local_config.has_calibration:
            t0 = time.perf_counter()

            # Normal mode: split the wide frame and detect in parallel
            sub_frames = [numpy.ascontiguousarray(s) for s in MultiCameraCapture.split_frame(image, num_cameras)]
            t_split = time.perf_counter()

            # Parallel detection only (C library releases GIL)
            detect_futures = [detection_pool.submit(fiducial_detectors[i].detect_fiducials, sub_frames[i], config)
                              for i in range(num_cameras)]

            # BGR conversion while detection runs
            if len(image.shape) == 2:
                display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            else:
                display_image = image.copy()
            t_cvt = time.perf_counter()

            sub_width = image.shape[1] // num_cameras

            # Collect detections, then pose estimate + publish sequentially
            all_detections = [f.result() for f in detect_futures]
            t_det = time.perf_counter()

            for cam_idx in range(num_cameras):
                cam_config = config.for_camera(cam_idx)
                if not cam_config.local_config.has_calibration:
                    output_publishers[cam_idx].send(cam_config, timestamp, None, fps if cam_idx == 0 else None)
                    continue

                raw_camera_servers[cam_idx].set_frame(sub_frames[cam_idx])

                image_observations = all_detections[cam_idx]
                display_sub = display_image[:, cam_idx * sub_width:(cam_idx + 1) * sub_width]
                [overlay_image_observation(display_sub, x) for x in image_observations]

                camera_pose_observation = camera_pose_estimator.solve_camera_pose(
                    image_observations, cam_config)

                output_publishers[cam_idx].send(cam_config, timestamp, camera_pose_observation, fps if cam_idx == 0 else None)
            t_pose = time.perf_counter()

            stream_server.set_frame(display_image)
            t_stream = time.perf_counter()

            if fps is not None:
                print(f"  cap:{(t_cap1-t_cap0)*1000:.1f} split:{(t_split-t0)*1000:.1f} cvt:{(t_cvt-t_split)*1000:.1f} "
                      f"det:{(t_det-t_split)*1000:.1f} pose:{(t_pose-t_det)*1000:.1f} "
                      f"stream:{(t_stream-t_pose)*1000:.1f} total:{(t_stream-t_cap0)*1000:.1f}ms")
            continue

        else:
            # No calibration
            _clear_aruco_outputs(output_publishers, config, timestamp)
            print("No calibration found")
            time.sleep(0.5)

        stream_server.set_frame(image)
