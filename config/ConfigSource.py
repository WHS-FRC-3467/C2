import json
from typing import Any

import cv2
import ntcore
import numpy

from config.config import ConfigStore, RemoteConfig
from type_defs import empty_float_array


class ConfigSource:
    def update(self, config_store: ConfigStore) -> None:
        raise NotImplementedError


class FileConfigSource(ConfigSource):
    CONFIG_FILENAME: str = "config.json"
    CALIBRATION_FILENAME: str = "calib_default.yml"

    @staticmethod
    def calibration_filename(camera_index: int) -> str:
        return f"calibration_{camera_index}.yml"

    def update(self, config_store: ConfigStore) -> None:
        # Get config
        with open(self.CONFIG_FILENAME, "r", encoding="utf-8") as config_file:
            config_data: dict[str, Any] = json.load(config_file)
            config_store.local_config.device_id = str(config_data["device_id"])
            config_store.local_config.server_ip = str(config_data["server_ip"])
            config_store.local_config.stream_port = int(config_data["stream_port"])
            config_store.local_config.num_cameras = int(
                config_data.get("num_cameras", 4)
            )
            config_store.local_config.enable_object_detection = bool(
                config_data.get("enable_object_detection", False)
            )

        # Load per-camera calibrations
        num_cameras = config_store.local_config.num_cameras
        config_store.local_config.camera_matrices = []
        config_store.local_config.distortion_coefficients_list = []
        config_store.local_config.has_calibrations = []

        for i in range(num_cameras):
            filename = self.calibration_filename(i)
            calibration_store = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
            camera_matrix = calibration_store.getNode("camera_matrix").mat()
            distortion_coefficients = calibration_store.getNode(
                "distortion_coefficients"
            ).mat()
            calibration_store.release()

            if isinstance(camera_matrix, numpy.ndarray) and isinstance(
                distortion_coefficients, numpy.ndarray
            ):
                config_store.local_config.camera_matrices.append(
                    numpy.asarray(camera_matrix, dtype=numpy.float64)
                )
                config_store.local_config.distortion_coefficients_list.append(
                    numpy.asarray(distortion_coefficients, dtype=numpy.float64)
                )
                config_store.local_config.has_calibrations.append(True)
            else:
                config_store.local_config.camera_matrices.append(empty_float_array())
                config_store.local_config.distortion_coefficients_list.append(
                    empty_float_array()
                )
                config_store.local_config.has_calibrations.append(False)

        # Legacy: has_calibration is True if any camera is calibrated
        config_store.local_config.has_calibration = any(
            config_store.local_config.has_calibrations
        )


class NTConfigSource(ConfigSource):
    def __init__(self) -> None:
        self._init_complete = False
        self._camera_id_sub: ntcore.StringSubscriber | None = None
        self._camera_resolution_width_sub: ntcore.IntegerSubscriber | None = None
        self._camera_resolution_height_sub: ntcore.IntegerSubscriber | None = None
        self._camera_exposure_sub: ntcore.IntegerSubscriber | None = None
        self._camera_gain_sub: ntcore.IntegerSubscriber | None = None
        self._fiducial_size_m_sub: ntcore.DoubleSubscriber | None = None
        self._tag_layout_sub: ntcore.StringSubscriber | None = None

    def update(self, config_store: ConfigStore) -> None:
        # Initialize subscribers on first call
        if not self._init_complete:
            nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
                "/" + config_store.local_config.device_id + "/config"
            )
            self._camera_id_sub = nt_table.getStringTopic("camera_id").subscribe(
                RemoteConfig.camera_id
            )
            self._camera_resolution_width_sub = nt_table.getIntegerTopic(
                "camera_resolution_width"
            ).subscribe(RemoteConfig.camera_resolution_width)
            self._camera_resolution_height_sub = nt_table.getIntegerTopic(
                "camera_resolution_height"
            ).subscribe(RemoteConfig.camera_resolution_height)
            self._camera_exposure_sub = nt_table.getIntegerTopic(
                "camera_exposure"
            ).subscribe(RemoteConfig.camera_exposure)
            self._camera_gain_sub = nt_table.getIntegerTopic("camera_gain").subscribe(
                RemoteConfig.camera_gain
            )
            self._fiducial_size_m_sub = nt_table.getDoubleTopic(
                "fiducial_size_m"
            ).subscribe(RemoteConfig.fiducial_size_m)
            self._tag_layout_sub = nt_table.getStringTopic("tag_layout").subscribe("")
            self._init_complete = True

        assert self._camera_id_sub is not None
        assert self._camera_resolution_width_sub is not None
        assert self._camera_resolution_height_sub is not None
        assert self._camera_exposure_sub is not None
        assert self._camera_gain_sub is not None
        assert self._fiducial_size_m_sub is not None
        assert self._tag_layout_sub is not None

        # Read config data
        config_store.remote_config.camera_id = self._camera_id_sub.get()
        config_store.remote_config.camera_resolution_width = (
            self._camera_resolution_width_sub.get()
        )
        config_store.remote_config.camera_resolution_height = (
            self._camera_resolution_height_sub.get()
        )
        config_store.remote_config.camera_exposure = self._camera_exposure_sub.get()
        config_store.remote_config.camera_gain = self._camera_gain_sub.get()
        config_store.remote_config.fiducial_size_m = self._fiducial_size_m_sub.get()
        try:
            config_store.remote_config.tag_layout = json.loads(
                self._tag_layout_sub.get()
            )
        except json.JSONDecodeError:
            config_store.remote_config.tag_layout = None
