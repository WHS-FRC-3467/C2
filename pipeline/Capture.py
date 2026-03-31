import dataclasses
import subprocess
import sys
import time
from typing import Tuple

import cv2
import numpy
from config.config import ConfigStore


class Capture:
    """Interface for receiving camera frames."""

    def __init__(self) -> None:
        raise NotImplementedError

    def get_frame(self, config_store: ConfigStore) -> Tuple[bool, cv2.Mat]:
        """Return the next frame from the camera."""
        raise NotImplementedError

    @classmethod
    def _config_changed(cls, config_a: ConfigStore, config_b: ConfigStore) -> bool:
        """Check if capture pipeline needs full restart (camera device or resolution changed)."""
        if config_a == None and config_b == None:
            return False
        if config_a == None or config_b == None:
            return True

        remote_a = config_a.remote_config
        remote_b = config_b.remote_config

        return remote_a.camera_id != remote_b.camera_id or remote_a.camera_resolution_width != remote_b.camera_resolution_width or remote_a.camera_resolution_height != remote_b.camera_resolution_height

    @classmethod
    def _exposure_changed(cls, config_a: ConfigStore, config_b: ConfigStore) -> bool:
        if config_a == None or config_b == None:
            return True
        return config_a.remote_config.camera_exposure != config_b.remote_config.camera_exposure or config_a.remote_config.camera_gain != config_b.remote_config.camera_gain


class DefaultCapture(Capture):
    """"Read from camera with default OpenCV config."""

    def __init__(self) -> None:
        self._video = None
        self._last_config = None

    def get_frame(self, config_store: ConfigStore) -> Tuple[bool, cv2.Mat]:
        if self._video is not None and self._config_changed(self._last_config, config_store):
            print("Config changed, stopping capture session")
            self._video.release()
            self._video = None
            time.sleep(2)

        if self._video is None:
            if config_store.remote_config.camera_id == "":
                print("No camera ID, waiting to start capture session")
            else:
                print("Starting default capture session")
                camera_id = config_store.remote_config.camera_id
                try:
                    camera_id = int(camera_id)
                except ValueError:
                    pass

                self._video = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
                if not self._video.isOpened():
                    self._video.release()
                    self._video = None
                    return False, cv2.Mat(numpy.ndarray([]))
                self._video.set(cv2.CAP_PROP_FRAME_WIDTH, config_store.remote_config.camera_resolution_width)
                self._video.set(cv2.CAP_PROP_FRAME_HEIGHT, config_store.remote_config.camera_resolution_height)
                self._video.set(cv2.CAP_PROP_EXPOSURE, config_store.remote_config.camera_exposure)
                self._video.set(cv2.CAP_PROP_GAIN, config_store.remote_config.camera_gain)
                print("Default capture session ready")
        elif self._exposure_changed(self._last_config, config_store):
            self._video.set(cv2.CAP_PROP_EXPOSURE, config_store.remote_config.camera_exposure)
            self._video.set(cv2.CAP_PROP_GAIN, config_store.remote_config.camera_gain)

        self._last_config = ConfigStore(dataclasses.replace(config_store.local_config),
                                        dataclasses.replace(config_store.remote_config))

        if self._video is not None:
            retval, image = self._video.read()
            if not retval:
                print("Capture session failed, restarting")
                self._video.release()
                self._video = None
                return False, cv2.Mat(numpy.ndarray([]))
            return retval, image
        else:
            return False, cv2.Mat(numpy.ndarray([]))


class GStreamerCapture(Capture):
    """"Read from camera with GStreamer."""

    def __init__(self) -> None:
        pass

    _video = None
    _last_config: ConfigStore

    def get_frame(self, config_store: ConfigStore) -> Tuple[bool, cv2.Mat]:
        if self._video != None and self._config_changed(self._last_config, config_store):
            print("Config changed, stopping capture session")
            self._video.release()
            self._video = None
            time.sleep(2)

        if self._video == None:
            if config_store.remote_config.camera_id == "":
                print("No camera ID, waiting to start capture session")
            else:
                print("Starting capture session")
                self._video = cv2.VideoCapture("v4l2src device=" + str(config_store.remote_config.camera_id) + " extra_controls=\"c,exposure_absolute=" + str(
                    config_store.remote_config.camera_exposure) + ",gain=" + str(config_store.remote_config.camera_gain) + ",sharpness=0,brightness=0\" ! image/jpeg,format=MJPG,width=" + str(config_store.remote_config.camera_resolution_width) + ",height=" + str(config_store.remote_config.camera_resolution_height) + " ! jpegdec ! video/x-raw ! appsink drop=1", cv2.CAP_GSTREAMER)
                print("Capture session ready")

        self._last_config = ConfigStore(dataclasses.replace(config_store.local_config),
                                        dataclasses.replace(config_store.remote_config))

        if self._video != None:
            retval, image = self._video.read()
            if not retval:
                print("Capture session failed, restarting")
                self._video.release()
                self._video = None  # Force reconnect
                sys.exit(1)
            return retval, image
        else:
            return False, cv2.Mat(numpy.ndarray([]))


class MultiCameraCapture(Capture):
    """Read from a multi-camera V4L2 module that outputs a single wide combined frame.

    Captures in grayscale by disabling OpenCV's RGB conversion. The full wide
    frame (width * num_cameras x height) is returned from get_frame(). Use
    split_frame() to divide it into individual camera sub-frames.
    """

    def __init__(self) -> None:
        self._video = None
        self._last_config = None
        self._needs_exposure_apply = False

    def get_frame(self, config_store: ConfigStore) -> Tuple[bool, cv2.Mat]:
        if self._video is not None and self._config_changed(self._last_config, config_store):
            print("Config changed, stopping capture session")
            self._video.release()
            self._video = None
            time.sleep(2)

        if self._video is None:
            if config_store.remote_config.camera_id == "":
                print("No camera ID, waiting to start capture session")
            else:
                num_cameras = config_store.local_config.num_cameras
                total_width = config_store.remote_config.camera_resolution_width * num_cameras
                height = config_store.remote_config.camera_resolution_height
                print(f"Starting multi-camera capture session ({num_cameras} cameras, {total_width}x{height})")

                camera_id = config_store.remote_config.camera_id
                try:
                    camera_id = int(camera_id)
                except ValueError:
                    pass
                self._video = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
                self._video.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'GREY'))
                self._video.set(cv2.CAP_PROP_CONVERT_RGB, 0)
                self._video.set(cv2.CAP_PROP_FRAME_WIDTH, total_width)
                self._video.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self._video.set(cv2.CAP_PROP_FPS, 45)
                self._needs_exposure_apply = True
                print("Multi-camera capture session ready")

        # Flag exposure update when NT values change
        if self._video is not None and self._exposure_changed(self._last_config, config_store):
            self._needs_exposure_apply = True

        self._last_config = ConfigStore(dataclasses.replace(config_store.local_config),
                                        dataclasses.replace(config_store.remote_config))

        if self._video is not None:
            retval, image = self._video.read()
            if not retval:
                print("Capture session failed, restarting")
                self._video.release()
                self._video = None
                sys.exit(1)

            # Apply exposure/gain via v4l2-ctl after stream is active
            if self._needs_exposure_apply:
                self._needs_exposure_apply = False
                device = str(config_store.remote_config.camera_id)
                self._v4l2_set(device, "exposure", config_store.remote_config.camera_exposure)
                self._v4l2_set(device, "analogue_gain", config_store.remote_config.camera_gain)

            return retval, image
        else:
            return False, cv2.Mat(numpy.ndarray([]))

    @staticmethod
    def _v4l2_set(device: str, ctrl: str, value: int) -> None:
        result = subprocess.run(
            ['v4l2-ctl', '-d', device, '--set-ctrl', f'{ctrl}={value}'],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  v4l2-ctl ERROR setting {ctrl}={value}: {result.stderr.strip()}")
        else:
            print(f"  v4l2-ctl: {ctrl}={value}")

    @staticmethod
    def split_frame(frame: cv2.Mat, num_cameras: int) -> list:
        """Split a wide combined frame into individual camera sub-frames.

        Returns numpy views into the original frame (zero-copy), so drawing
        on sub-frames will also appear on the original.
        """
        total_width = frame.shape[1]
        sub_width = total_width // num_cameras
        return [frame[:, i * sub_width:(i + 1) * sub_width] for i in range(num_cameras)]
