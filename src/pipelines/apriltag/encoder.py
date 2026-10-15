"""Build the apriltag pipeline's flatbuffer message (schema/apriltag.fbs)."""

from __future__ import annotations

import flatbuffers
from dsv0 import CameraObservation as CO
from dsv0 import CameraOutput as Out
from dsv0 import Frame as F
from dsv0 import PoseSolution as PS
from dsv0.Pose3d import CreatePose3d

from pipelines.apriltag.pose_solver import CameraPoseObservation, PoseSolution

TYPE_STRING = "dsv0_fb"


def encode_camera_frame(
    camera_index: int, observation: CameraPoseObservation | None, fps: int
) -> bytes:
    """One ``Frame`` holding one ``CameraOutput``. The capture time travels as the NT value timestamp."""
    builder = flatbuffers.Builder(256)

    observation_offset = None
    if observation is not None:
        observation_offset = _build_observation(builder, observation)

    Out.Start(builder)
    Out.AddCameraIndex(builder, camera_index)
    if observation_offset is not None:
        Out.AddCameraObservation(builder, observation_offset)
    Out.AddFps(builder, fps)
    camera_output = Out.End(builder)

    F.StartCamerasVector(builder, 1)
    builder.PrependUOffsetTRelative(camera_output)
    cameras = builder.EndVector()

    F.Start(builder)
    F.AddCameras(builder, cameras)
    builder.Finish(F.End(builder))
    return bytes(builder.Output())


def _build_observation(
    builder: flatbuffers.Builder, observation: CameraPoseObservation
) -> int:
    best = _build_solution(builder, observation.best)
    alternate = (
        _build_solution(builder, observation.alternate)
        if observation.alternate
        else None
    )

    CO.StartTagIdsVector(builder, len(observation.tag_ids))
    for tag_id in reversed(observation.tag_ids):
        builder.PrependInt32(tag_id)
    tag_ids = builder.EndVector()

    CO.Start(builder)
    CO.AddSolution0(builder, best)
    if alternate is not None:
        CO.AddSolution1(builder, alternate)
    CO.AddTagIds(builder, tag_ids)
    return CO.End(builder)


def _build_solution(builder: flatbuffers.Builder, solution: PoseSolution) -> int:
    t = solution.field_to_camera.translation()
    q = solution.field_to_camera.rotation().getQuaternion()
    PS.Start(builder)
    # Structs must be created right before they are added to their table.
    PS.AddPose(
        builder, CreatePose3d(builder, t.X(), t.Y(), t.Z(), q.W(), q.X(), q.Y(), q.Z())
    )
    PS.AddError(builder, solution.reprojection_error)
    return PS.End(builder)
