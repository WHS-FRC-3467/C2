import math
import sys
import os
from typing import List, SupportsFloat, SupportsIndex, Union

import flatbuffers
import ntcore
from config.config import ConfigStore
from vision_types import CameraPoseObservation

# Add schema to path for generated flatbuffer modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "schema"))

from dsv0.Pose3d import CreatePose3d
from dsv0.PoseSolution import (
    PoseSolutionStart,
    PoseSolutionAddPose,
    PoseSolutionAddError,
    PoseSolutionEnd,
)
from dsv0.CameraObservation import (
    CameraObservationStart,
    CameraObservationAddSolution0,
    CameraObservationAddSolution1,
    CameraObservationAddTagIds,
    CameraObservationStartTagIdsVector,
    CameraObservationEnd,
)
from dsv0.CameraOutput import (
    CameraOutputStart,
    CameraOutputAddTimestampUs,
    CameraOutputAddCameraIndex,
    CameraOutputAddCameraObservation,
    CameraOutputAddFps,
    CameraOutputEnd,
)
from dsv0.Frame import (
    FrameStart,
    FrameAddTimestampUs,
    FrameAddCameras,
    FrameStartCamerasVector,
    FrameEnd,
)


class OutputPublisher:
    def send(
        self,
        config_store: ConfigStore,
        timestamp: float,
        observation: Union[CameraPoseObservation, None],
        fps: Union[int, None] = None,
    ) -> None:
        raise NotImplementedError


class NTOutputPublisher(OutputPublisher):
    _init_complete: bool = False
    _observations_pub: ntcore.DoubleArrayPublisher
    _fps_pub: ntcore.IntegerPublisher

    def __init__(self, camera_index: int = -1) -> None:
        self._camera_index = camera_index

    def send(
        self,
        config_store: ConfigStore,
        timestamp: float,
        observation: Union[CameraPoseObservation, None],
        fps: Union[int, None] = None,
    ) -> None:
        if not self._init_complete:
            self._init_complete = True
            if self._camera_index >= 0:
                table_path = (
                    "/"
                    + config_store.local_config.device_id
                    + "/output/camera_"
                    + str(self._camera_index)
                )
            else:
                table_path = "/" + config_store.local_config.device_id + "/output"
            nt_table = ntcore.NetworkTableInstance.getDefault().getTable(table_path)
            self._observations_pub = nt_table.getDoubleArrayTopic(
                "observations"
            ).publish(
                ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True)
            )
            self._fps_pub = nt_table.getIntegerTopic("fps").publish()

        if fps is not None:
            self._fps_pub.set(fps)
        observation_data: List[SupportsFloat | SupportsIndex] = [0.0]
        if observation is not None:
            observation_data[0] = 1
            observation_data.append(observation.error_0)
            observation_data.append(observation.pose_0.translation().X())
            observation_data.append(observation.pose_0.translation().Y())
            observation_data.append(observation.pose_0.translation().Z())
            observation_data.append(observation.pose_0.rotation().getQuaternion().W())
            observation_data.append(observation.pose_0.rotation().getQuaternion().X())
            observation_data.append(observation.pose_0.rotation().getQuaternion().Y())
            observation_data.append(observation.pose_0.rotation().getQuaternion().Z())
            if observation.error_1 is not None and observation.pose_1 is not None:
                observation_data[0] = 2
                observation_data.append(observation.error_1)
                observation_data.append(observation.pose_1.translation().X())
                observation_data.append(observation.pose_1.translation().Y())
                observation_data.append(observation.pose_1.translation().Z())
                observation_data.append(
                    observation.pose_1.rotation().getQuaternion().W()
                )
                observation_data.append(
                    observation.pose_1.rotation().getQuaternion().X()
                )
                observation_data.append(
                    observation.pose_1.rotation().getQuaternion().Y()
                )
                observation_data.append(
                    observation.pose_1.rotation().getQuaternion().Z()
                )
            for tag_id in observation.tag_ids:
                observation_data.append(tag_id)
        self._observations_pub.set(observation_data, math.floor(timestamp * 1000000))


def _build_pose_solution(builder, pose, error):
    """Build a PoseSolution table. Returns the offset."""
    t = pose.translation()
    q = pose.rotation().getQuaternion()
    pose_struct = CreatePose3d(builder, t.X(), t.Y(), t.Z(), q.W(), q.X(), q.Y(), q.Z())
    PoseSolutionStart(builder)
    PoseSolutionAddPose(builder, pose_struct)
    PoseSolutionAddError(builder, error)
    return PoseSolutionEnd(builder)


class NTFlatbufferOutputPublisher(OutputPublisher):
    """Publishes observations as flatbuffer-encoded byte arrays over NetworkTables."""

    _init_complete: bool = False
    _frame_pub: ntcore.RawPublisher
    _fps_pub: ntcore.IntegerPublisher

    def __init__(self, camera_index: int = -1) -> None:
        self._camera_index = camera_index

    def send(
        self,
        config_store: ConfigStore,
        timestamp: float,
        observation: Union[CameraPoseObservation, None],
        fps: Union[int, None] = None,
    ) -> None:
        if not self._init_complete:
            self._init_complete = True
            if self._camera_index >= 0:
                table_path = (
                    "/"
                    + config_store.local_config.device_id
                    + "/output/camera_"
                    + str(self._camera_index)
                )
            else:
                table_path = "/" + config_store.local_config.device_id + "/output"
            nt_table = ntcore.NetworkTableInstance.getDefault().getTable(table_path)
            self._frame_pub = nt_table.getRawTopic("observation").publish(
                "dsv0_fb",
                ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True),
            )
            self._fps_pub = nt_table.getIntegerTopic("fps").publish()

        if fps is not None:
            self._fps_pub.set(fps)

        timestamp_us = math.floor(timestamp * 1000000)
        builder = flatbuffers.Builder(256)

        # Build camera observation if present
        cam_obs_offset = None
        if observation is not None:
            # Build solutions
            sol0 = _build_pose_solution(
                builder, observation.pose_0, observation.error_0
            )
            sol1 = None
            if observation.pose_1 is not None and observation.error_1 is not None:
                sol1 = _build_pose_solution(
                    builder, observation.pose_1, observation.error_1
                )

            # Build tag_ids vector
            CameraObservationStartTagIdsVector(builder, len(observation.tag_ids))
            for tag_id in reversed(observation.tag_ids):
                builder.PrependInt32(tag_id)
            tag_ids_vec = builder.EndVector()

            # Build CameraObservation
            CameraObservationStart(builder)
            CameraObservationAddSolution0(builder, sol0)
            if sol1 is not None:
                CameraObservationAddSolution1(builder, sol1)
            CameraObservationAddTagIds(builder, tag_ids_vec)
            cam_obs_offset = CameraObservationEnd(builder)

        # Build CameraOutput
        CameraOutputStart(builder)
        CameraOutputAddTimestampUs(builder, timestamp_us)
        CameraOutputAddCameraIndex(builder, max(self._camera_index, 0))
        if cam_obs_offset is not None:
            CameraOutputAddCameraObservation(builder, cam_obs_offset)
        if fps is not None:
            CameraOutputAddFps(builder, fps)
        cam_output = CameraOutputEnd(builder)

        # Wrap in Frame
        FrameStartCamerasVector(builder, 1)
        builder.PrependUOffsetTRelative(cam_output)
        cameras_vec = builder.EndVector()

        FrameStart(builder)
        FrameAddTimestampUs(builder, timestamp_us)
        FrameAddCameras(builder, cameras_vec)
        frame = FrameEnd(builder)

        builder.Finish(frame)
        buf = builder.Output()

        self._frame_pub.set(bytes(buf), timestamp_us)
