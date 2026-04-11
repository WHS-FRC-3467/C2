import math
import os
import sys
from typing import List

import flatbuffers
import ntcore

from vision_types import ObjectDetectionObservation

# Add schema to path for generated flatbuffer modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "schema"))

from objectdetections.Detection import (
    DetectionAddAreaPx,
    DetectionAddCentroidX,
    DetectionAddCentroidY,
    DetectionAddClassId,
    DetectionAddConfidence,
    DetectionAddPitchDeg,
    DetectionAddX0,
    DetectionAddX1,
    DetectionAddY0,
    DetectionAddY1,
    DetectionAddYawDeg,
    DetectionEnd,
    DetectionStart,
)
from objectdetections.DetectionFrame import (
    DetectionFrameAddDetections,
    DetectionFrameEnd,
    DetectionFrameStart,
    DetectionFrameStartDetectionsVector,
)


def _build_detection(builder: flatbuffers.Builder, detection: ObjectDetectionObservation) -> int:
    DetectionStart(builder)
    DetectionAddClassId(builder, detection.class_id)
    DetectionAddConfidence(builder, detection.confidence)
    DetectionAddX0(builder, detection.x0)
    DetectionAddY0(builder, detection.y0)
    DetectionAddX1(builder, detection.x1)
    DetectionAddY1(builder, detection.y1)
    DetectionAddCentroidX(builder, detection.centroid_x)
    DetectionAddCentroidY(builder, detection.centroid_y)
    DetectionAddPitchDeg(builder, detection.pitch_deg)
    DetectionAddYawDeg(builder, detection.yaw_deg)
    DetectionAddAreaPx(builder, detection.area_px)
    return DetectionEnd(builder)


def _build_detections_vector(builder: flatbuffers.Builder, detection_offsets: List[int]) -> int:
    DetectionFrameStartDetectionsVector(builder, len(detection_offsets))
    for detection_offset in reversed(detection_offsets):
        builder.PrependUOffsetTRelative(detection_offset)
    return builder.EndVector()


class NTObjectDetectionPublisher:
    _init_complete: bool = False
    _frame_pub: ntcore.RawPublisher

    def __init__(self, device_id: str, table_name: str = "object_detection") -> None:
        self._device_id = device_id
        self._table_name = table_name

    def send(self, timestamp: float, detections: List[ObjectDetectionObservation]) -> None:
        if not self._init_complete:
            nt_table = ntcore.NetworkTableInstance.getDefault().getTable(
                f"/{self._device_id}/{self._table_name}"
            )
            self._frame_pub = nt_table.getRawTopic("detections").publish(
                "objectdetections_fb",
                ntcore.PubSubOptions(periodic=0, sendAll=True, keepDuplicates=True),
            )
            self._init_complete = True

        timestamp_us = math.floor(timestamp * 1000000)
        builder = flatbuffers.Builder(256)
        detection_offsets = [_build_detection(builder, detection) for detection in detections]
        detections_vec = _build_detections_vector(builder, detection_offsets)

        DetectionFrameStart(builder)
        DetectionFrameAddDetections(builder, detections_vec)
        frame = DetectionFrameEnd(builder)
        builder.Finish(frame)

        self._frame_pub.set(bytes(builder.Output()), timestamp_us)
