import dataclasses
from dataclasses import dataclass, field
from typing import Any

from type_defs import FloatArray, empty_float_array


@dataclass
class LocalConfig:
    device_id: str = ""
    server_ip: str = ""
    stream_port: int = 8000
    num_cameras: int = 4
    enable_object_detection: bool = False
    has_calibration: bool = False
    camera_matrix: FloatArray = field(default_factory=empty_float_array)
    distortion_coefficients: FloatArray = field(default_factory=empty_float_array)
    # Per-camera calibration data
    camera_matrices: list[FloatArray] = field(default_factory=list)
    distortion_coefficients_list: list[FloatArray] = field(default_factory=list)
    has_calibrations: list[bool] = field(default_factory=list)


@dataclass
class RemoteConfig:
    camera_id: str = ""
    camera_resolution_width: int = 0
    camera_resolution_height: int = 0
    camera_exposure: int = 0
    camera_gain: int = 0
    fiducial_size_m: float = 0
    tag_layout: Any | None = None


@dataclass
class ConfigStore:
    local_config: LocalConfig
    remote_config: RemoteConfig

    def for_camera(self, index: int) -> "ConfigStore":
        """Return a ConfigStore with calibration data for a specific camera index.

        This allows pipeline code to use config_store.local_config.camera_matrix
        transparently without knowing about the multi-camera setup.
        """
        local = dataclasses.replace(
            self.local_config,
            camera_matrix=self.local_config.camera_matrices[index]
            if index < len(self.local_config.camera_matrices)
            else empty_float_array(),
            distortion_coefficients=self.local_config.distortion_coefficients_list[
                index
            ]
            if index < len(self.local_config.distortion_coefficients_list)
            else empty_float_array(),
            has_calibration=self.local_config.has_calibrations[index]
            if index < len(self.local_config.has_calibrations)
            else False,
        )
        return ConfigStore(local, self.remote_config)
