"""Draws the board at a target pose, as the user sees it through the lens.

Adapted from Pose Calib by Pavel Rojtberg (AGPL-3.0, see LICENSE).
"""

from __future__ import annotations

import cv2
import numpy

from calibration.board import Board
from calibration.distvis import make_distort_map

_PIXELS_PER_SQUARE = 24
# The distortion map is built at this size, then the result is scaled up.
_WORK_SIZE = (640, 480)


class BoardPreview:
    def __init__(self, board: Board) -> None:
        image = board.board.generateImage(
            (
                board.spec.squares_x * _PIXELS_PER_SQUARE,
                board.spec.squares_y * _PIXELS_PER_SQUARE,
            )
        )
        # The board image is in pixels; the board's own frame is in meters.
        self._meters_per_pixel = board.spec.square_m / _PIXELS_PER_SQUARE
        styled = image.copy()
        styled[styled == 0] = 64  # dark gray instead of black
        self._image = cv2.cvtColor(styled, cv2.COLOR_GRAY2BGR)
        self._image[:, :, 0::2] = 0  # green
        self._mask = numpy.ones(image.shape, dtype=numpy.uint8)
        self._size: tuple[int, int] = _WORK_SIZE
        self._k_work = numpy.eye(3)
        self._maps: numpy.ndarray | None = None
        """(2, height, width): x and y sample maps for cv2.remap."""

    def set_camera(
        self, k: numpy.ndarray, distortion: numpy.ndarray | None, size: tuple[int, int]
    ) -> None:
        if distortion is None:
            distortion = numpy.zeros(4)
        self._size = size
        scale = numpy.diag((_WORK_SIZE[0] / size[0], _WORK_SIZE[1] / size[1], 1))
        k = scale @ k
        self._k_work = cv2.getOptimalNewCameraMatrix(k, distortion, _WORK_SIZE, 1)[0]
        self._maps = make_distort_map(k, _WORK_SIZE, distortion, self._k_work)

    def project(
        self, rvec: numpy.ndarray, tvec: numpy.ndarray, mask: bool = False
    ) -> numpy.ndarray:
        """The board (or, with ``mask``, its silhouette as 0/1) at pose (rvec, tvec), full camera size."""
        assert self._maps is not None, "call set_camera() first"
        rotation = cv2.Rodrigues(numpy.asarray(rvec, dtype=numpy.float64))[0]
        # Homography from board-image pixels to undistorted work-image pixels.
        homography = self._k_work @ numpy.column_stack(
            [rotation[:, 0], rotation[:, 1], tvec]
        )
        homography = homography @ numpy.diag(
            [self._meters_per_pixel, self._meters_per_pixel, 1]
        )
        homography /= homography[2, 2]
        source = self._mask if mask else self._image
        warped = cv2.warpPerspective(
            source, homography, _WORK_SIZE, flags=cv2.INTER_NEAREST
        )
        warped = cv2.remap(warped, self._maps[0], self._maps[1], cv2.INTER_NEAREST)
        return cv2.resize(warped, self._size, interpolation=cv2.INTER_NEAREST)
