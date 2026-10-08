from dataclasses import dataclass
from typing import Optional

import ntcore
from wpimath.geometry import Pose3d

from input.Config import ConfigStore

@dataclass
class TimestampedPose:
    pose: Pose3d
    timestamp_us: int


class RobotPoseSource:
    def get_pose(self) -> Optional[TimestampedPose]:
        raise NotImplementedError


class NTRobotPoseSource(RobotPoseSource):
    _pose_sub: ntcore.StructSubscriber

    def __init__(self, config: ConfigStore) -> None:
        nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
                                "/" + config.local_config.device_id + "/Pose"
                            )
        self._pose_sub = nt_table.getStructTopic("Pose", Pose3d).subscribe(None)

    def get_pose(self) -> Optional[TimestampedPose]:
        timestamped_struct = self._pose_sub.getAtomic()
        if timestamped_struct.value is None:
            return None

        return TimestampedPose(timestamped_struct.value, timestamped_struct.time)