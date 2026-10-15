"""One interactive calibration of one camera, driven from the dashboard.

    idle -> collecting -> solving -> solved -> (save) -> idle
                 ^           |
                 |           v
                 +------- failed / "needs more data"

Collecting runs guidance on every NEW frame of the chosen camera, at full
quality (no JPEG). When guidance says there is enough data (or the user asks),
mrcal solves in the background. Saving writes calibration_<i>.yml and the
AprilTag pipeline uses it from its next frame.
"""

from __future__ import annotations

import dataclasses
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy

from calibration import mrcal_solver
from calibration.board import Board, BoardSpec
from calibration.guidance import Guidance
from calibration.mrcal_solver import MrcalError, MrcalResult
from config.calibration_store import CalibrationStore
from config.local_config import CameraCalibration
from core.frames import FrameSet

log = logging.getLogger(__name__)

# A result is "good" when all of these hold (pixels). mrcal's uncertainty is
# 1-sigma at infinity; with a lean lens model it is somewhat optimistic.
GOOD_RMS_PX = 1.0
GOOD_MEDIAN_UNCERTAINTY_PX = 0.5
GOOD_WORST_UNCERTAINTY_PX = 1.5
# After "continue collecting", gather at least this many more keyframes before solving again.
_MORE_KEYFRAMES = 5
_IDLE_WAIT_S = 0.1


@dataclass
class _Solve:
    result: MrcalResult | None = None
    error: str | None = None


class CalibrationSession:
    def __init__(
        self, num_cameras: int, store: CalibrationStore, board_path: Path
    ) -> None:
        self._num_cameras = num_cameras
        self._store = store
        self._board_path = board_path
        self.board_spec = BoardSpec.load(board_path)

        self._lock = threading.Condition()
        self._phase = "idle"
        self._camera = 0
        self._board: Board | None = None
        self._guidance: Guidance | None = None
        self._frames: FrameSet | None = None
        self._last_sequence = -1
        self._force_capture = False
        self._solve_at_keyframes = 0
        self._solve = _Solve()
        self._message = ""
        self._detected = 0
        self._overlay: tuple[int, numpy.ndarray] | None = None
        self._saved_path: str | None = None
        threading.Thread(target=self._run, name="calibration", daemon=True).start()

    # ----------------------------------------------------- dashboard actions

    def start(self, camera: int, spec: BoardSpec) -> None:
        if not 0 <= camera < self._num_cameras:
            raise ValueError(f"No camera {camera}")
        spec.validate()
        mrcal_solver.check_installed()  # fail before the user spends time collecting
        self._board_path.write_text(spec.to_json())
        with self._lock:
            self.board_spec = spec
            self._board = Board(spec)
            self._camera = camera
            self._guidance = (
                None  # created on the first frame, which gives the image size
            )
            self._solve = _Solve()
            self._solve_at_keyframes = 0
            self._saved_path = None
            self._message = "Waiting for frames..."
            self._phase = "collecting"
            self._lock.notify_all()
        log.info("Calibration of camera %d started (%s)", camera, spec)

    def stop(self) -> None:
        with self._lock:
            self._phase = "idle"
            self._guidance = None
            self._overlay = None
            self._message = ""

    def capture_now(self) -> None:
        """Capture the next frame as a keyframe, wherever the board is."""
        with self._lock:
            self._force_capture = True

    def solve_now(self) -> None:
        with self._lock:
            guidance = self._guidance
            if (
                self._phase not in ("collecting", "failed", "solved")
                or guidance is None
            ):
                raise ValueError("Nothing to solve yet")
            if len(guidance.keyframes) < 4:
                raise ValueError("Need at least 4 keyframes")
            self._begin_solve()

    def continue_collecting(self) -> None:
        with self._lock:
            if self._guidance is None:
                raise ValueError("No calibration in progress")
            self._guidance.resume()
            self._solve_at_keyframes = len(self._guidance.keyframes) + _MORE_KEYFRAMES
            self._phase = "collecting"
            self._message = f"Collect at least {_MORE_KEYFRAMES} more keyframes."

    def save(self) -> str:
        with self._lock:
            result = self._solve.result
            if self._phase != "solved" or result is None:
                raise ValueError("Nothing solved to save")
            camera = self._camera
        calibration = CameraCalibration(
            result.camera_matrix,
            result.distortion,
            result.image_size[0],
            result.image_size[1],
            result.rms_error_px,
        )
        path = self._store.save(
            camera,
            calibration,
            {
                "solver": f"mrcal {mrcal_solver.LENSMODEL}",
                "keyframes": len(self._guidance.keyframes) if self._guidance else 0,
                "worst_reprojection_error": result.worst_error_px,
                "uncertainty_median_px": result.uncertainty_px["median"],
                "uncertainty_worst_px": result.uncertainty_px["worst"],
            },
        )
        with self._lock:
            self._saved_path = str(path)
            self._phase = "idle"
            self._guidance = None
            self._overlay = None
            self._message = f"Saved {path.name}; the AprilTag pipeline is using it now."
        return str(path)

    # ------------------------------------------------- capture loop / views

    def submit(self, frames: FrameSet) -> None:
        """Called by the capture loop for every frame set. Cheap when idle."""
        with self._lock:
            if self._phase == "collecting":
                self._frames = frames
                self._lock.notify_all()

    def latest_overlay(self) -> tuple[int, numpy.ndarray] | None:
        with self._lock:
            return self._overlay

    def snapshot(self) -> dict:
        with self._lock:
            guidance = self._guidance
            result = self._solve.result
            return {
                "phase": self._phase,
                "camera": self._camera,
                "message": self._message,
                "board": dataclasses.asdict(self.board_spec),
                "board_corners_detected": self._detected,
                "board_corners_total": self.board_spec.num_corners,
                "guidance": dataclasses.asdict(guidance.status()) if guidance else None,
                "result": _result_summary(result) if result else None,
                "error": self._solve.error,
                "saved_path": self._saved_path,
                "previous": _calibration_summary(self._store.all()[self._camera]),
            }

    # --------------------------------------------------------------- private

    def _begin_solve(self) -> None:
        """Caller holds the lock."""
        assert self._guidance is not None
        keyframes = list(self._guidance.keyframes)
        spec, size = self.board_spec, self._guidance.image_size
        focal = float(self._guidance.calibrator.k[0, 0])
        self._phase = "solving"
        self._message = f"Solving with mrcal ({len(keyframes)} keyframes)..."
        self._solve = _Solve()

        def run() -> None:
            try:
                result = mrcal_solver.solve(keyframes, spec, size, focal)
                solve, message = _Solve(result=result), _verdict(result)
                phase = "solved"
            except (MrcalError, OSError, ValueError) as e:
                log.exception("mrcal solve failed")
                solve, message, phase = (
                    _Solve(error=str(e)),
                    "mrcal failed; see the error below.",
                    "failed",
                )
            with self._lock:
                if self._phase == "solving":  # not stopped meanwhile
                    self._solve, self._message, self._phase = solve, message, phase

        threading.Thread(target=run, name="mrcal-solve", daemon=True).start()

    def _run(self) -> None:
        while True:
            with self._lock:
                while (
                    self._frames is None or self._frames.sequence == self._last_sequence
                ):
                    self._lock.wait(_IDLE_WAIT_S)
                frames, board = self._frames, self._board
                camera, force = self._camera, self._force_capture
                self._last_sequence = frames.sequence
                self._force_capture = False
                if self._phase != "collecting" or board is None:
                    continue
            try:
                self._process(frames, board, camera, force)
            except Exception:  # noqa: BLE001 - keep the session alive
                log.exception("Calibration frame failed")

    def _process(
        self, frames: FrameSet, board: Board, camera: int, force: bool
    ) -> None:
        image = frames.cameras[camera]
        size = (image.shape[1], image.shape[0])
        with self._lock:
            if self._guidance is None or self._guidance.image_size != size:
                self._guidance = Guidance(board, size)
            guidance = self._guidance
        observation = board.detect(image)
        guidance.update(observation, frames.sequence, force)
        overlay = _draw(image, guidance, observation)
        with self._lock:
            if self._guidance is not guidance or self._phase != "collecting":
                return
            self._detected = observation.count if observation else 0
            self._message = guidance.last_event
            self._overlay = (frames.sequence, overlay)
            if (
                guidance.converged
                and len(guidance.keyframes) >= self._solve_at_keyframes
            ):
                self._begin_solve()


def _draw(image: numpy.ndarray, guidance: Guidance, observation) -> numpy.ndarray:
    out = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    # Markers sized for the image, so they stay visible when the stream is scaled down.
    radius = max(3, out.shape[1] // 300)
    # Where the user should hold the board (translucent green).
    if guidance.target_image is not None:
        mask = guidance.target_image.any(axis=2)
        out[mask] = (0.5 * out[mask] + 0.5 * guidance.target_image[mask]).astype(
            numpy.uint8
        )
    # Coverage so far: every captured corner (mrcal wants the whole image covered, corners too).
    for keyframe in guidance.keyframes:
        for x, y in keyframe.observation.image_points:
            cv2.circle(out, (int(x), int(y)), radius, (0, 200, 255), -1)
    # What is detected now.
    if observation is not None:
        for x, y in observation.image_points:
            cv2.circle(
                out, (int(x), int(y)), 2 * radius, (255, 120, 0), max(2, radius // 2)
            )
    return out


def _verdict(result: MrcalResult) -> str:
    u = result.uncertainty_px
    if (
        result.rms_error_px <= GOOD_RMS_PX
        and u["median"] <= GOOD_MEDIAN_UNCERTAINTY_PX
        and u["worst"] <= GOOD_WORST_UNCERTAINTY_PX
    ):
        return "Good calibration. Review and save."
    return "Usable but uncertain: collect more data (cover the image corners, tilt the board more) or save anyway."


def _result_summary(result: MrcalResult) -> dict:
    k = result.camera_matrix
    return {
        "fx": k[0, 0],
        "fy": k[1, 1],
        "cx": k[0, 2],
        "cy": k[1, 2],
        "distortion": result.distortion.ravel().tolist(),
        "image_size": list(result.image_size),
        "rms_error_px": result.rms_error_px,
        "worst_error_px": result.worst_error_px,
        "outliers": result.outliers,
        "points": result.points,
        "uncertainty_px": {
            key: v for key, v in result.uncertainty_px.items() if key != "grid"
        },
        "board_warp_mm": (
            [w * 1000 for w in result.board_warp_m] if result.board_warp_m else None
        ),
        "good": _verdict(result).startswith("Good"),
    }


def _calibration_summary(calibration: CameraCalibration | None) -> dict | None:
    if calibration is None:
        return None
    k = calibration.camera_matrix
    return {
        "fx": float(k[0, 0]),
        "fy": float(k[1, 1]),
        "cx": float(k[0, 2]),
        "cy": float(k[1, 2]),
        "image_size": [calibration.image_width, calibration.image_height],
        "rms_error_px": calibration.reprojection_error,
    }
