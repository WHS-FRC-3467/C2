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
from output.OutputPublisher import NTFlatbufferOutputPublisher, OutputPublisher
from output.overlay_util import *
from output.CalibrationUploadServer import CalibrationUploadServer
from output.StreamServer import MjpegServer, RawCameraMjpegServer
from output.RawFrameRecorder import RawFrameRecorder
from pipeline.CameraPoseEstimator import MultiTargetCameraPoseEstimator
from pipeline.Capture import DefaultCapture, MultiCameraCapture
from pipeline.ArucoNanoDetector import ArucoNanoFiducialDetector
from pipeline.FiducialDetector import ArucoFiducialDetector


def _load_objdetect_camera_matrix(calibration_file: str) -> numpy.ndarray:
    calibration_store = cv2.FileStorage(calibration_file, cv2.FILE_STORAGE_READ)
    camera_matrix = calibration_store.getNode("camera_matrix").mat()
    calibration_store.release()
    if type(camera_matrix) == numpy.ndarray:
        return camera_matrix
    return numpy.array([])


def _run_aruco(config, local_config_source, remote_config_source, calibration_command_source):
    num_cameras = config.local_config.num_cameras

    capture = MultiCameraCapture()
    fiducial_detectors = [ArucoNanoFiducialDetector(cv2.aruco.DICT_APRILTAG_36h11) for _ in range(num_cameras)]
    camera_pose_estimator = MultiTargetCameraPoseEstimator()
    output_publishers = [NTFlatbufferOutputPublisher(i) for i in range(num_cameras)]
    stream_server = MjpegServer()
    raw_camera_servers = [RawCameraMjpegServer(i) for i in range(num_cameras)]
    calibration_session = CalibrationSession()
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
    with RawFrameRecorder(config.local_config.record_raw_frames) as recorder:
        while True:
            remote_config_source.update(config)
            timestamp = time.time()
            t_cap0 = time.perf_counter()
            success, image = capture.get_frame(config)
            t_cap1 = time.perf_counter()
            if not success:
                time.sleep(0.5)
                continue

            recorder.submit(image)

            fps = None
            frame_count += 1
            if time.time() - last_print > 1:
                last_print = time.time()
                fps = frame_count
                print("Running at", frame_count, "fps")
                frame_count = 0

            if calibration_command_source.get_calibrating(config):
                was_calibrating = True
                calibration_session.process_frame(image, calibration_command_source.get_capture_flag(config))

            elif was_calibrating:
                calibration_session.finish()
                sys.exit(0)

            elif config.local_config.has_calibration:
                t0 = time.perf_counter()

                sub_frames = [numpy.ascontiguousarray(s) for s in MultiCameraCapture.split_frame(image, num_cameras)]
                t_split = time.perf_counter()

                detect_futures = [detection_pool.submit(fiducial_detectors[i].detect_fiducials, sub_frames[i], config)
                                  for i in range(num_cameras)]

                if len(image.shape) == 2:
                    display_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                else:
                    display_image = image.copy()
                t_cvt = time.perf_counter()

                sub_width = image.shape[1] // num_cameras

                all_detections = [f.result() for f in detect_futures]
                t_det = time.perf_counter()

                for cam_idx in range(num_cameras):
                    cam_config = config.for_camera(cam_idx)
                    if not cam_config.local_config.has_calibration:
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
                print("No calibration found")
                time.sleep(0.5)

            stream_server.set_frame(image)


def _run_object_detection(config, local_config_source, remote_config_source):
    from output.ObjectDetectionPublisher import NTObjectDetectionPublisher
    from pipeline.TensorRTYoloDetector import TensorRTYoloDetector

    camera_matrix = _load_objdetect_camera_matrix(config.local_config.objdetect_calibration_file)
    if camera_matrix.size == 0:
        print(f"ERROR: No calibration found at {config.local_config.objdetect_calibration_file}")
        print("Object detection requires a calibrated camera. Exiting.")
        sys.exit(1)

    capture = DefaultCapture()
    detector = TensorRTYoloDetector(
        model_path=config.local_config.objdetect_model_path,
        camera_matrix=camera_matrix,
    )
    publisher = NTObjectDetectionPublisher(config.local_config.device_id)
    stream_server = MjpegServer()
    stream_server.start(config)
    raw_camera_server = RawCameraMjpegServer(0)
    raw_camera_server.start(config.local_config.stream_port + 1)
    print(f"Raw camera stream at port {config.local_config.stream_port + 1}")

    frame_count = 0
    last_print = 0
    with RawFrameRecorder(config.local_config.record_raw_frames) as recorder:
        while True:
            remote_config_source.update(config)
            timestamp = time.time()
            t_cap0 = time.perf_counter()
            success, image = capture.get_frame(config)
            t_cap1 = time.perf_counter()
            if not success:
                time.sleep(0.5)
                continue

            recorder.submit(image)

            fps = None
            frame_count += 1
            if time.time() - last_print > 1:
                last_print = time.time()
                fps = frame_count
                print("Running at", frame_count, "fps")
                frame_count = 0

            t_det0 = time.perf_counter()
            detections = detector.detect(image)
            t_det1 = time.perf_counter()

            publisher.send(timestamp, detections)

            raw_camera_server.set_frame(image)
            display_image = image if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for det in detections:
                overlay_object_detection(display_image, det)
            stream_server.set_frame(display_image)
            t_stream = time.perf_counter()

            if fps is not None:
                print(f"  cap:{(t_cap1-t_cap0)*1000:.1f} det:{(t_det1-t_det0)*1000:.1f} "
                      f"stream:{(t_stream-t_det1)*1000:.1f} total:{(t_stream-t_cap0)*1000:.1f}ms "
                      f"detections:{len(detections)}")


if __name__ == "__main__":
    config = ConfigStore(LocalConfig(), RemoteConfig())
    local_config_source: ConfigSource = FileConfigSource()
    remote_config_source: ConfigSource = NTConfigSource()
    calibration_command_source: CalibrationCommandSource = NTCalibrationCommandSource()

    local_config_source.update(config)

    ntcore.NetworkTableInstance.getDefault().setServer(config.local_config.server_ip)
    ntcore.NetworkTableInstance.getDefault().startClient4(config.local_config.device_id)

    calibration_upload_server = CalibrationUploadServer(port=config.local_config.stream_port + 10)
    calibration_upload_server.start()

    if config.local_config.detector_mode == "object_detection":
        print("Starting in object detection mode")
        _run_object_detection(config, local_config_source, remote_config_source)
    else:
        print("Starting in aruco detection mode")
        _run_aruco(config, local_config_source, remote_config_source, calibration_command_source)
