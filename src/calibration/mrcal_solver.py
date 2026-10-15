"""The final calibration solve, done by mrcal (https://mrcal.secretsauce.net).

mrcal is required: install it with ``sudo apt install mrcal``. It isn't in
C2's virtualenv (it is only packaged for the system Python), so C2 runs
mrcal's command-line solver and a small report script as separate processes.

Compared to OpenCV's calibrateCamera, mrcal:
- models the board's deformation (printed boards are never perfectly flat),
- rejects outlier corners,
- reports projection uncertainty across the image, so we know whether the
  data was good enough.

The lens model is LENSMODEL_OPENCV8: the same 8-coefficient model OpenCV
uses, so the result plugs straight into C2's solvePnP.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy

from calibration.board import BoardSpec
from calibration.estimator import Keyframe

LENSMODEL = "LENSMODEL_OPENCV8"
SYSTEM_PYTHON = "/usr/bin/python3"
_REPORT_SCRIPT = Path(__file__).with_name("mrcal_report.py")
_SOLVE_TIMEOUT_S = 300


class MrcalError(RuntimeError):
    pass


@dataclass(frozen=True)
class MrcalResult:
    camera_matrix: numpy.ndarray
    distortion: numpy.ndarray
    """8 coefficients: k1, k2, p1, p2, k3, k4, k5, k6 (OpenCV order)."""
    image_size: tuple[int, int]
    rms_error_px: float
    worst_error_px: float
    outliers: int
    points: int
    uncertainty_px: dict
    """center, median, worst (1-sigma, pixels, for points at infinity), and the sampled grid."""
    board_warp_m: list[float] | None
    log: str


def check_installed() -> None:
    """Raise MrcalError with install instructions if mrcal isn't usable."""
    if shutil.which("mrcal-calibrate-cameras") is None:
        raise MrcalError(
            "mrcal is not installed. Install it with: sudo apt install mrcal"
        )
    probe = subprocess.run(
        [SYSTEM_PYTHON, "-c", "import mrcal"], capture_output=True, text=True
    )
    if probe.returncode != 0:
        raise MrcalError(
            f"mrcal's Python module failed to import: {probe.stderr.strip()[-300:]}"
        )


def solve(
    keyframes: list[Keyframe],
    spec: BoardSpec,
    image_size: tuple[int, int],
    focal_guess_px: float,
) -> MrcalResult:
    check_installed()
    if len(keyframes) < 4:
        raise MrcalError("Need at least 4 keyframes")
    with tempfile.TemporaryDirectory(prefix="c2-mrcal-") as tmp:
        directory = Path(tmp)
        _write_corners(directory / "corners.vnl", keyframes, spec)
        command = [
            "mrcal-calibrate-cameras",
            "--corners-cache", "corners.vnl",
            "--lensmodel", LENSMODEL,
            "--focal", f"{focal_guess_px:.1f}",
            "--imagersize", str(image_size[0]), str(image_size[1]),
            "--object-spacing", repr(spec.square_m),
            "--object-width-n", str(spec.corners_x),
            "--object-height-n", str(spec.corners_y),
            "--outdir", ".",
            "frame*.png",
        ]  # fmt: skip
        solved = subprocess.run(
            command,
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=_SOLVE_TIMEOUT_S,
        )
        output = solved.stdout + solved.stderr
        if solved.returncode != 0:
            raise MrcalError(f"mrcal-calibrate-cameras failed:\n{output[-2000:]}")
        models = sorted(directory.glob("*.cameramodel"))
        if not models:
            raise MrcalError(f"mrcal wrote no camera model:\n{output[-2000:]}")

        reported = subprocess.run(
            [SYSTEM_PYTHON, str(_REPORT_SCRIPT), str(models[0])],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if reported.returncode != 0:
            raise MrcalError(f"mrcal report failed:\n{reported.stderr[-2000:]}")
        report = json.loads(reported.stdout)

    if report["lensmodel"] != LENSMODEL:
        raise MrcalError(f"Unexpected lens model {report['lensmodel']}")
    fx, fy, cx, cy, *distortion = report["intrinsics"]
    supplied = sum(kf.observation.count for kf in keyframes)
    return MrcalResult(
        camera_matrix=numpy.array(
            [[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=numpy.float64
        ),
        distortion=numpy.array([distortion], dtype=numpy.float64),
        image_size=(int(report["image_size"][0]), int(report["image_size"][1])),
        rms_error_px=report["rms_error_px"],
        worst_error_px=report["worst_error_px"],
        outliers=supplied - report["inliers"],
        points=supplied,
        uncertainty_px=report["uncertainty_px"],
        board_warp_m=report["board_warp_m"],
        log=output,
    )


def _write_corners(path: Path, keyframes: list[Keyframe], spec: BoardSpec) -> None:
    """mrcal's corners file: one row per board corner per image, row-major, '- - -' if unseen.

    ChArUco corner ids are already row-major over the inner-corner grid.
    """
    lines = ["# filename x y level"]
    for index, keyframe in enumerate(keyframes):
        name = f"frame{index:04d}.png"
        seen = dict(
            zip(
                keyframe.observation.corner_ids.tolist(),
                keyframe.observation.image_points.tolist(),
            )
        )
        for corner in range(spec.num_corners):
            if corner in seen:
                x, y = seen[corner]
                lines.append(f"{name} {x:.4f} {y:.4f} 0")
            else:
                lines.append(f"{name} - - -")
    path.write_text("\n".join(lines) + "\n")
