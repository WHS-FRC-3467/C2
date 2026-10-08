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
from dsv0.CameraOutput import (
    CameraOutputStart,
    CameraOutputAddCameraIndex,
    CameraOutputAddCameraObservation,
    CameraOutputAddFps,
    CameraOutputEnd,
)
from dsv0.PerCameraResults import (
    PerCameraResultsStartResultsVector,
    PerCameraResultsStart,
    PerCameraResultsAddResults,
    PerCameraResultsEnd,
)
from dsv0.Frame import (
    FrameStart,
    FrameAddResultsType,
    FrameAddResults,
    FrameEnd,
)


def _build_pose_solution(builder, pose, error):
    """Build a PoseSolution table. Returns the offset."""
    t = pose.translation()
    q = pose.rotation().getQuaternion()
    pose_struct = CreatePose3d(builder, t.X(), t.Y(), t.Z(), q.W(), q.X(), q.Y(), q.Z())
    PoseSolutionStart(builder)
    PoseSolutionAddPose(builder, pose_struct)
    PoseSolutionAddReprojectionError(builder, error)
    return PoseSolutionEnd(builder)


class NTFlatbufferOutputPublisher:
    """Publishes observations as flatbuffer-encoded byte arrays over NetworkTables."""

    _init_complete: bool = False
    _frame_pub: ntcore.RawPublisher
    _fps_pub: ntcore.IntegerPublisher

    def send(
        self,
        config_store: ConfigStore,
        timestamp: float,
        observations: list[Optional[CameraPoseObservation]],
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

        # Build camera observation if present
        camera_outputs = []
        for i, observation in enumerate(observations):
            if observation is None:
                # Build CameraOutput w/o observation
                CameraOutputStart(builder)
                CameraOutputAddCameraIndex(builder, i)
                if fps is not None:
                    CameraOutputAddFps(builder, fps)
                camera_outputs.append(CameraOutputEnd(builder))
                continue

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
            CameraOutputAddCameraIndex(builder, i)
            if cam_obs_offset is not None:
                CameraOutputAddCameraObservation(builder, cam_obs_offset)
            if fps is not None:
                CameraOutputAddFps(builder, fps)
            camera_outputs.append(CameraOutputEnd(builder))

        # Wrap in PerCameraResults
        PerCameraResultsStartResultsVector(builder, len(camera_outputs))
        for cam_output in reversed(camera_outputs):
            builder.PrependUOffsetTRelative(cam_output)
        cameras_vec = builder.EndVector()

        PerCameraResultsStart(builder)
        PerCameraResultsAddResults(builder, cameras_vec)
        per_camera_results = PerCameraResultsEnd(builder)

        FrameStart(builder)
        FrameAddResultsType(builder, Results.Results().PerCameraResults)
        FrameAddResults(builder, per_camera_results)
        frame = FrameEnd(builder)

        builder.Finish(frame)
        buf = builder.Output()

        self._frame_pub.set(bytes(buf), timestamp_us)
