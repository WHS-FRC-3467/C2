"""The ChArUco calibration board: its geometry, detection, and printable image."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy

log = logging.getLogger(__name__)

# Dictionaries offered in the UI, by OpenCV name.
DICTIONARIES = ("DICT_4X4_1000", "DICT_5X5_1000", "DICT_6X6_1000", "DICT_7X7_1000")


@dataclass(frozen=True)
class BoardSpec:
    """A ChArUco board. Defaults match charuco_board.png in this repo."""

    squares_x: int = 12
    squares_y: int = 9
    square_m: float = 0.030
    """Measured edge length of one square on the PRINTED board, meters."""
    marker_m: float = 0.022
    """Measured edge length of one marker on the printed board, meters."""
    dictionary: str = "DICT_5X5_1000"

    @property
    def corners_x(self) -> int:
        """Inner (chessboard) corners per row."""
        return self.squares_x - 1

    @property
    def corners_y(self) -> int:
        return self.squares_y - 1

    @property
    def num_corners(self) -> int:
        return self.corners_x * self.corners_y

    def validate(self) -> None:
        if self.squares_x < 3 or self.squares_y < 3:
            raise ValueError("The board needs at least 3x3 squares")
        if not 0 < self.marker_m < self.square_m:
            raise ValueError(
                "Marker size must be positive and smaller than the square size"
            )
        if self.dictionary not in DICTIONARIES:
            raise ValueError(f"Unknown dictionary {self.dictionary}")
        markers_needed = self.squares_x * self.squares_y // 2
        dictionary_size = int(self.dictionary.rsplit("_", 1)[1])
        if markers_needed > dictionary_size:
            raise ValueError(
                f"{self.dictionary} has too few markers for a {self.squares_x}x{self.squares_y} board"
            )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def load(path: Path) -> BoardSpec:
        if not path.exists():
            return BoardSpec()
        try:
            spec = BoardSpec(**json.loads(path.read_text()))
            spec.validate()
            return spec
        except (ValueError, TypeError) as e:
            log.warning("Ignoring %s (%s); using the default board", path, e)
            return BoardSpec()


@dataclass(frozen=True)
class BoardObservation:
    """The board's inner corners found in one image."""

    corner_ids: numpy.ndarray
    """Shape (N,), int: row-major index into the corners_x x corners_y grid."""
    image_points: numpy.ndarray
    """Shape (N, 2), float32 pixels."""
    object_points: numpy.ndarray
    """Shape (N, 3), float32 meters, on the board plane (z = 0)."""

    @property
    def count(self) -> int:
        return len(self.corner_ids)


class Board:
    def __init__(self, spec: BoardSpec) -> None:
        spec.validate()
        self.spec = spec
        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, spec.dictionary)
        )
        self.board = cv2.aruco.CharucoBoard(
            (spec.squares_x, spec.squares_y), spec.square_m, spec.marker_m, dictionary
        )
        self._detector = cv2.aruco.CharucoDetector(self.board)
        self._all_corners = numpy.asarray(
            self.board.getChessboardCorners(), dtype=numpy.float32
        )

    @property
    def size_m(self) -> numpy.ndarray:
        """(width, height, 0) of the inner-corner grid, meters."""
        return (
            numpy.array([self.spec.corners_x, self.spec.corners_y, 0])
            * self.spec.square_m
        )

    def detect(self, gray: numpy.ndarray) -> BoardObservation | None:
        corners, ids, _, _ = self._detector.detectBoard(gray)
        if ids is None or len(ids) == 0:
            return None
        ids = ids.ravel().astype(int)
        return BoardObservation(
            ids, corners.reshape(-1, 2).astype(numpy.float32), self._all_corners[ids]
        )

    def image(
        self, pixels_per_square: int = 200, margin_squares: float = 0.5
    ) -> numpy.ndarray:
        """The board as a printable grayscale image, with a white margin."""
        margin = int(pixels_per_square * margin_squares)
        size = (
            self.spec.squares_x * pixels_per_square + 2 * margin,
            self.spec.squares_y * pixels_per_square + 2 * margin,
        )
        return self.board.generateImage(size, marginSize=margin)

    def estimate_pose(
        self,
        observation: BoardObservation,
        camera_matrix: numpy.ndarray,
        distortion: numpy.ndarray | None,
    ) -> tuple[numpy.ndarray, numpy.ndarray] | None:
        """Board pose in the camera frame (rvec, tvec), or None."""
        if observation.count < 6:
            return None
        ok, rvec, tvec = cv2.solvePnP(
            observation.object_points,
            observation.image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        return (rvec.ravel(), tvec.ravel()) if ok else None
