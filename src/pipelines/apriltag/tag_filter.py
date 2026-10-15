"""Drop tags that are too small in the image before solving.

Corner detection error is roughly a fixed number of pixels, so a tag only a
few dozen pixels across gives noisy corners, and a solve that relies on such
tags can be off by tens of centimeters (or pick the wrong pose entirely for a
single tag). Size in pixels is what matters, which makes one limit mean the
same thing on every camera: a longer focal length sees the same tag bigger,
so it keeps using tags from farther away. Approximately:

    distance at the limit ≈ focal_length_px * tag_size_m / min_tag_size_px

A tag's size is its longest edge, the least foreshortened when the tag is
seen at an angle.

In simulation (focal length ~914 px, 2025 field, 2 m/s), 50 px (about 3 m
there) removed the large outliers: multi-tag max error 76 cm -> under 7 cm,
single-tag median 45 cm -> ~5 cm. 38 px (~4 m) left a 25 cm worst case.
A low limit (25 px, ~6 m) was worse than none: it removed far tags from
otherwise good multi-tag solves.
"""

from __future__ import annotations

import numpy

from pipelines.apriltag.detector import TagDetection


def tag_size_px(detection: TagDetection) -> float:
    """The tag's longest edge in the image, pixels."""
    edges = numpy.linalg.norm(
        detection.corners - numpy.roll(detection.corners, 1, axis=0), axis=1
    )
    return float(edges.max())


def exclude_small_tags(
    detections: list[TagDetection], min_size_px: float
) -> tuple[list[TagDetection], list[TagDetection]]:
    """Split detections into (used, excluded). ``min_size_px`` <= 0 disables the filter."""
    if min_size_px <= 0:
        return detections, []
    used, excluded = [], []
    for detection in detections:
        (used if tag_size_px(detection) >= min_size_px else excluded).append(detection)
    return used, excluded
