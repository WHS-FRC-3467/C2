"""Summarize an mrcal camera model as JSON. Run by the SYSTEM Python, where mrcal lives.

    /usr/bin/python3 mrcal_report.py camera-0.cameramodel

Not imported by C2: mrcal is installed with apt, built against the system
Python and NumPy, which C2's virtualenv doesn't share. Keep this file
self-contained (standard library, numpy, mrcal only).
"""

import json
import sys

import mrcal
import numpy as np

# Uncertainty is sampled on this grid across the imager.
GRID = (16, 12)


def main(path: str) -> None:
    model = mrcal.cameramodel(path)
    lensmodel, intrinsics = model.intrinsics()
    width, height = (int(v) for v in model.imagersize())

    # Projection uncertainty: how far (pixels, 1-sigma, worst direction) a point at
    # infinity could project, given the noise in the calibration data.
    q = mrcal.sample_imager(GRID[0], GRID[1], width, height)
    v = mrcal.unproject(q, lensmodel, intrinsics)
    uncertainty = mrcal.projection_uncertainty(
        v, model, atinfinity=True, what="worstdirection-stdev"
    )
    center = mrcal.projection_uncertainty(
        mrcal.unproject(
            np.array([(width - 1) / 2, (height - 1) / 2]), lensmodel, intrinsics
        ),
        model,
        atinfinity=True,
        what="worstdirection-stdev",
    )

    inputs = model.optimization_inputs()
    warp = inputs.get("calobject_warp")
    # Reprojection errors of every inlier corner (outliers already removed by mrcal).
    errors = np.linalg.norm(mrcal.measurements_board(inputs), axis=-1)

    json.dump(
        {
            "lensmodel": lensmodel,
            "intrinsics": [float(x) for x in intrinsics],
            "image_size": [width, height],
            "uncertainty_px": {
                "center": float(center),
                "median": float(np.median(uncertainty)),
                "worst": float(np.max(uncertainty)),
                "grid": np.asarray(uncertainty).tolist(),
            },
            "board_warp_m": [float(x) for x in warp] if warp is not None else None,
            "inliers": int(errors.size),
            "rms_error_px": float(np.sqrt(np.mean(errors**2))) if errors.size else None,
            "worst_error_px": float(np.max(errors)) if errors.size else None,
        },
        sys.stdout,
    )


if __name__ == "__main__":
    main(sys.argv[1])
