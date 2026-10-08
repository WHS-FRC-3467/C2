import dataclasses
from dataclasses import dataclass, field
from typing import Any, List

import numpy
import numpy.typing
from wpimath.geometry import Pose3d


@dataclass
class LocalConfig:
    device_id: str = ""
    server_ip: str = ""
    stream_port: int = 8000
    num_cameras: int = 4
    camera_id: str = ""
    camera_resolution_width: int = 0
    camera_resolution_height: int = 0
    camera_matrices: List = field(default_factory=list)
    distortion_coefficients: List = field(default_factory=list)


@dataclass
class RemoteConfig:
    camera_exposure: int = 0
    camera_gain: int = 0
    camera_extrinsics: list[Pose3d] = []
    fiducial_size_m: float = 0
    tag_layout: Any = None


@dataclass
class ConfigStore:
    local_config: LocalConfig
    remote_config: RemoteConfig
