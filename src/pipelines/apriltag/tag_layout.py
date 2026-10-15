"""Parse WPILib's AprilTagFieldLayout JSON (the format AprilTagFieldLayout.serialize() writes)."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy
from wpimath.geometry import Pose3d, Quaternion, Rotation3d, Transform3d, Translation3d

from vision.coordinate_systems import wpilib_translation_to_opencv


@dataclass(frozen=True)
class FieldTag:
    pose: Pose3d
    """Field-to-tag pose."""
    corners_opencv: numpy.ndarray
    """The tag's four corners in field coordinates, OpenCV axes, in the
    detector's corner order. Shape (4, 3)."""


def parse_tag_layout(layout_json: str, tag_size_m: float) -> dict[int, FieldTag]:
    """Return tags by ID. Raises ValueError on malformed JSON."""
    try:
        layout = json.loads(layout_json)
        return {int(tag["ID"]): _field_tag(_pose_from_json(tag["pose"]), tag_size_m) for tag in layout["tags"]}
    except (KeyError, TypeError, json.JSONDecodeError) as e:
        raise ValueError(f"Invalid tag layout: {e}") from e


def _pose_from_json(pose: dict) -> Pose3d:
    t = pose["translation"]
    q = pose["rotation"]["quaternion"]
    return Pose3d(Translation3d(t["x"], t["y"], t["z"]), Rotation3d(Quaternion(q["W"], q["X"], q["Y"], q["Z"])))


def _field_tag(pose: Pose3d, size_m: float) -> FieldTag:
    half = size_m / 2.0
    # (tag Y, tag Z) of each corner, in the order aruco_nano reports them for
    # a real FRC tag: bottom-right, bottom-left, top-left, top-right, as seen
    # facing the tag (tag +Y is the viewer's right). That is OpenCV's TL, TR,
    # BR, BL order turned 180 degrees, because OpenCV's AprilTag dictionary is
    # rotated 180 degrees from the official tag images.
    # tests/test_apriltag_pipeline.py checks this with rendered official tags.
    offsets = [(half, -half), (-half, -half), (-half, half), (half, half)]
    corners = [
        wpilib_translation_to_opencv((pose + Transform3d(Translation3d(0, y, z), Rotation3d())).translation())
        for y, z in offsets
    ]
    return FieldTag(pose, numpy.array(corners, dtype=numpy.float64))
