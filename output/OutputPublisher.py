import math
import sys
import os
from typing import Optional, Union

import flatbuffers
import ntcore
from config.Config import ConfigStore
from vision_types import CameraPoseObservation

# Add schema to path for generated flatbuffer modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "schema"))

from dsv0 import Results
from dsv0.Pose3d import CreatePose3d
from dsv0.PoseSolution import (
    PoseSolutionAddReprojectionError,
    PoseSolutionStart,
    PoseSolutionAddPose,
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
from dsv0.CombinedResults import (
    CombinedResultsStart,
    CombinedResultsAddRobotObservation,
    CombinedResultsAddFps,
    CombinedResultsEnd,
)
from dsv0.Frame import (
    FrameStart,
    FrameAddResultsType,
    FrameAddResults,
    FrameEnd,
)


def _build_camera_observation(builder, observation):
    """Serialize one pose, its pixel error, optional alternate, and used tag IDs.

    The schema reuses CameraObservation for a robot pose result; the containing
    CombinedResults table names it robot_observation to make its meaning clear.
    """
    t = observation.pose_0.translation()
    q = observation.pose_0.rotation().getQuaternion()
    pose_struct = CreatePose3d(builder, t.X(), t.Y(), t.Z(), q.W(), q.X(), q.Y(), q.Z())
    PoseSolutionStart(builder)
    PoseSolutionAddPose(builder, pose_struct)
    PoseSolutionAddReprojectionError(builder, observation.error_0)
    solution_0 = PoseSolutionEnd(builder)

    solution_1 = None
    if observation.pose_1 is not None and observation.error_1 is not None:
        t = observation.pose_1.translation()
        q = observation.pose_1.rotation().getQuaternion()
        pose_struct = CreatePose3d(
            builder, t.X(), t.Y(), t.Z(), q.W(), q.X(), q.Y(), q.Z()
        )
        PoseSolutionStart(builder)
        PoseSolutionAddPose(builder, pose_struct)
        PoseSolutionAddReprojectionError(builder, observation.error_1)
        solution_1 = PoseSolutionEnd(builder)

    CameraObservationStartTagIdsVector(builder, len(observation.tag_ids))
    for tag_id in reversed(observation.tag_ids):
        builder.PrependInt32(tag_id)
    tag_ids_vec = builder.EndVector()

    CameraObservationStart(builder)
    CameraObservationAddSolution0(builder, solution_0)
    if solution_1 is not None:
        CameraObservationAddSolution1(builder, solution_1)
    CameraObservationAddTagIds(builder, tag_ids_vec)
    return CameraObservationEnd(builder)


class NTFlatbufferOutputPublisher:
    """Publish one combined robot-pose frame as FlatBuffer bytes over NetworkTables."""

    _init_complete: bool = False
    _frame_pub: ntcore.RawPublisher
    _fps_pub: ntcore.IntegerPublisher

    def send(
        self,
        config_store: ConfigStore,
        timestamp: float,
        observation: Optional[CameraPoseObservation],
        fps: Union[int, None] = None,
    ) -> None:
        if not self._init_complete:
            self._init_complete = True
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

        robot_observation = (
            _build_camera_observation(builder, observation)
            if observation is not None
            else None
        )

        CombinedResultsStart(builder)
        if robot_observation is not None:
            CombinedResultsAddRobotObservation(builder, robot_observation)
        if fps is not None:
            CombinedResultsAddFps(builder, fps)
        combined_results = CombinedResultsEnd(builder)

        FrameStart(builder)
        FrameAddResultsType(builder, Results.Results().CombinedResults)
        FrameAddResults(builder, combined_results)
        frame = FrameEnd(builder)

        builder.Finish(frame)
        buf = builder.Output()

        self._frame_pub.set(bytes(buf), timestamp_us)
