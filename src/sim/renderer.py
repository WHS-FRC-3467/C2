"""Render what a camera would see of the field's AprilTags.

Only tags are drawn, on a textured background. That is everything the
AprilTag pipeline looks at.

The pipeline is: project each tag (with its white border) through a
distortion-free pinhole camera, then warp the whole image through the lens
distortion, so straight lines bend the way they do through a real lens.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy
from wpimath.geometry import Pose3d

from pipelines.apriltag.tag_layout import FieldTag

TAG_BLACK = 25
TAG_WHITE = 225
_CELL_PX = 24  # texture resolution of the tag images
_NEAR_PLANE_M = 0.05

# A 36h11 tag is 8 cells across the black border (that edge is the "tag size")
# plus a 1-cell white margin on each side, 10 cells in all.
_BORDER_SCALE = 10 / 8


@dataclass(frozen=True)
class CameraModel:
    """What the simulated camera really is (which may differ from its calibration)."""

    width: int
    height: int
    camera_matrix: numpy.ndarray
    distortion: numpy.ndarray


class TagRenderer:
    def __init__(self) -> None:
        self._tag_images: dict[int | tuple[int, int], numpy.ndarray] = {}
        self._distortion_maps: dict[tuple, tuple[numpy.ndarray, numpy.ndarray]] = {}
        self._backgrounds: dict[tuple[int, int, int], numpy.ndarray] = {}

    def render(
        self,
        camera: CameraModel,
        field_to_camera: Pose3d,
        tags: dict[int, FieldTag],
        tag_size_m: float,
        background_seed: int,
    ) -> numpy.ndarray:
        """A grayscale uint8 image of the tags as seen from ``field_to_camera``."""
        image = self._background(camera.width, camera.height, background_seed).copy()
        corners = _project_tags(camera, field_to_camera, tags, tag_size_m)
        # Far tags first, so nearer tags cover them.
        for tag_id, image_corners, _ in sorted(corners, key=lambda c: -c[2]):
            self._draw_tag(
                image, self._tag_texture(tag_id, image_corners), image_corners
            )
        if numpy.any(camera.distortion):
            map_x, map_y = self._distortion_map(camera)
            image = cv2.remap(
                image, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
            )
        return image

    # ---------------------------------------------------------------- caches

    def _tag_texture(self, tag_id: int, corners: numpy.ndarray) -> numpy.ndarray:
        """The tag image, pre-shrunk to about twice its on-screen size.

        Warping a large texture down to a few pixels point-samples it
        (aliasing), which jitters the corners of distant tags. A real sensor
        averages the light over each pixel, which this approximates (mipmapping).
        """
        full = self._tag_image(tag_id)
        on_screen = float(
            numpy.linalg.norm(numpy.roll(corners, 1, axis=0) - corners, axis=1).max()
        )
        size = full.shape[0]
        while size // 2 >= 2 * on_screen and size // 2 >= 20:
            size //= 2
        if size == full.shape[0]:
            return full
        key = (tag_id, size)
        if key not in self._tag_images:
            self._tag_images[key] = cv2.resize(
                full, (size, size), interpolation=cv2.INTER_AREA
            )
        return self._tag_images[key]

    def _tag_image(self, tag_id: int) -> numpy.ndarray:
        if tag_id not in self._tag_images:
            dictionary = cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_APRILTAG_36h11
            )
            marker = cv2.aruco.generateImageMarker(dictionary, tag_id, 8 * _CELL_PX)
            # OpenCV's AprilTag images are the official (printed) tags turned
            # 180 degrees; see pipelines/apriltag/tag_layout.py.
            marker = cv2.rotate(marker, cv2.ROTATE_180)
            marker = cv2.copyMakeBorder(
                marker,
                _CELL_PX,
                _CELL_PX,
                _CELL_PX,
                _CELL_PX,
                cv2.BORDER_CONSTANT,
                value=(255,),
            )
            self._tag_images[tag_id] = numpy.where(
                marker > 127, TAG_WHITE, TAG_BLACK
            ).astype(numpy.uint8)
        return self._tag_images[tag_id]

    def _background(self, width: int, height: int, seed: int) -> numpy.ndarray:
        """Smooth, low-contrast texture. A real camera never sees a perfectly flat image
        (and core/status.py treats a perfectly flat image as a dead camera)."""
        key = (width, height, seed)
        if key not in self._backgrounds:
            rng = numpy.random.default_rng(seed)
            coarse = rng.normal(
                0, 1, (max(2, height // 64), max(2, width // 64))
            ).astype(numpy.float32)
            smooth = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
            smooth = cv2.GaussianBlur(smooth, (0, 0), 8)
            smooth = 105 + 25 * smooth / max(1e-6, float(numpy.abs(smooth).max()))
            self._backgrounds[key] = numpy.clip(smooth, 0, 255).astype(numpy.uint8)
        return self._backgrounds[key]

    def _distortion_map(
        self, camera: CameraModel
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """For each pixel of the distorted output, where to sample the undistorted render."""
        key = (
            camera.width,
            camera.height,
            camera.camera_matrix.tobytes(),
            camera.distortion.tobytes(),
        )
        if key not in self._distortion_maps:
            xs, ys = numpy.meshgrid(
                numpy.arange(camera.width, dtype=numpy.float32),
                numpy.arange(camera.height, dtype=numpy.float32),
            )
            pixels = numpy.stack([xs, ys], axis=-1).reshape(-1, 1, 2)
            criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 1e-4)
            undistorted = cv2.undistortPointsIter(
                pixels,
                camera.camera_matrix,
                camera.distortion,
                numpy.eye(3),
                camera.camera_matrix,
                criteria,
            ).reshape(camera.height, camera.width, 2)
            self._distortion_maps[key] = (
                numpy.ascontiguousarray(undistorted[..., 0]),
                numpy.ascontiguousarray(undistorted[..., 1]),
            )
        return self._distortion_maps[key]

    @staticmethod
    def _draw_tag(
        image: numpy.ndarray, tag_image: numpy.ndarray, corners: numpy.ndarray
    ) -> None:
        """Warp ``tag_image`` onto ``image`` with its corners at ``corners`` (TL, TR, BR, BL)."""
        height, width = image.shape
        x0, y0 = numpy.floor(corners.min(axis=0)).astype(int)
        x1, y1 = numpy.ceil(corners.max(axis=0)).astype(int) + 1
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width), min(y1, height)
        if x1 <= x0 or y1 <= y0:
            return
        # Only warp into the tag's bounding box, not the whole frame.
        size = tag_image.shape[0]
        src = numpy.array([[0, 0], [size, 0], [size, size], [0, size]], numpy.float32)
        dst = (corners - [x0, y0]).astype(numpy.float32)
        homography = cv2.getPerspectiveTransform(src, dst)
        box = (x1 - x0, y1 - y0)
        warped = cv2.warpPerspective(tag_image, homography, box, flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(
            numpy.full_like(tag_image, 255), homography, box, flags=cv2.INTER_LINEAR
        )
        region = image[y0:y1, x0:x1]
        alpha = mask.astype(numpy.float32) / 255.0  # anti-aliased tag edges
        region[:] = (warped * alpha + region * (1 - alpha)).astype(numpy.uint8)


def _project_tags(
    camera: CameraModel,
    field_to_camera: Pose3d,
    tags: dict[int, FieldTag],
    tag_size_m: float,
) -> list[tuple[int, numpy.ndarray, float]]:
    """(tag id, pixel corners TL/TR/BR/BL of the bordered tag, distance) for each visible tag."""
    rotation = field_to_camera.rotation().toMatrix()
    position = numpy.array(
        [field_to_camera.X(), field_to_camera.Y(), field_to_camera.Z()]
    )
    k = camera.camera_matrix
    half = tag_size_m / 2 * _BORDER_SCALE
    # Tag-frame (y, z) of each corner as seen facing the tag (tag +Y is the
    # viewer's right): TL, TR, BR, BL.
    offsets = numpy.array(
        [[0, -half, half], [0, half, half], [0, half, -half], [0, -half, -half]]
    )

    visible = []
    for tag_id, tag in tags.items():
        tag_rotation = tag.pose.rotation().toMatrix()
        tag_position = numpy.array([tag.pose.X(), tag.pose.Y(), tag.pose.Z()])
        # Back-face culling: the tag's +X axis points out of its face.
        if numpy.dot(tag_rotation[:, 0], position - tag_position) <= 0:
            continue
        corners_field = tag_position + offsets @ tag_rotation.T
        # Field -> camera (WPILib axes) -> OpenCV axes (+X right, +Y down, +Z forward).
        in_camera = (corners_field - position) @ rotation
        opencv = numpy.stack(
            [-in_camera[:, 1], -in_camera[:, 2], in_camera[:, 0]], axis=1
        )
        if numpy.any(opencv[:, 2] < _NEAR_PLANE_M):
            continue
        pixels = numpy.stack(
            [
                k[0, 0] * opencv[:, 0] / opencv[:, 2] + k[0, 2],
                k[1, 1] * opencv[:, 1] / opencv[:, 2] + k[1, 2],
            ],
            axis=1,
        )
        # Skip tags entirely off-screen (with a margin for distortion).
        margin = 0.25 * max(camera.width, camera.height)
        if (
            pixels[:, 0].max() < -margin
            or pixels[:, 0].min() > camera.width + margin
            or pixels[:, 1].max() < -margin
            or pixels[:, 1].min() > camera.height + margin
        ):
            continue
        visible.append((tag_id, pixels, float(math.dist(position, tag_position))))
    return visible
