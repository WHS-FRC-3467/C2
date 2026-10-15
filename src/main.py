"""C2 entry point: capture from the multi-camera module and run pipelines.

    python src/main.py [--config config.json]     (run from the repo root)
    python src/main.py --sim                       (render cameras from the robot sim's true pose)

The flow is:

    MultiCameraCapture --FrameSet--> PipelineManager --> one thread per pipeline --> NT (flatbuffers)
                                 \\-> DebugStreams --> DashboardServer (web page on web_port)

See README.md for the NetworkTables layout.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
import time
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
# The flatc-generated modules (schema/<namespace>/) import each other by namespace.
sys.path.insert(0, str(REPO_DIR / "schema"))

import ntcore  # noqa: E402
import numpy  # noqa: E402

from calibration.session import CalibrationSession  # noqa: E402
from capture.capture import Capture  # noqa: E402
from capture.multi_camera_capture import (
    CaptureSettings,
    MultiCameraCapture,
)  # noqa: E402
from config.calibration_store import CalibrationStore  # noqa: E402
from config.local_config import (
    CameraCalibration,
    LocalConfig,
    load_local_config,
)  # noqa: E402
from core.log_buffer import LogBuffer  # noqa: E402
from core.nt_settings import NTSettings  # noqa: E402
from core.status import StatusPublisher  # noqa: E402
from core.time_sync import TimeSync  # noqa: E402
from pipelines import PIPELINE_TYPES  # noqa: E402
from pipelines.pipeline import PipelineContext  # noqa: E402
from pipelines.runner import PipelineManager, PipelineRunner  # noqa: E402
from servers.dashboard_server import DashboardServer  # noqa: E402
from servers.debug_streams import DebugStreams  # noqa: E402
from sim.simulated_capture import SimulatedCapture, default_intrinsics  # noqa: E402

log = logging.getLogger("c2")


def connect_nt(config: LocalConfig) -> tuple[ntcore.NetworkTableInstance, TimeSync]:
    inst = ntcore.NetworkTableInstance.getDefault()
    inst.setServer(config.server_ip)
    inst.startClient4(config.device_id)
    time_sync = TimeSync(
        inst,
        server_address=config.server_ip,
        client_name=f"{config.device_id}-timesync",
        probe_interval_s=config.time_sync_probe_interval_s,
        diagnostics_table=f"/{config.device_id}/status/time_sync",
    )
    time_sync.start()
    return inst, time_sync


def build_pipelines(
    config: LocalConfig,
    inst: ntcore.NetworkTableInstance,
    calibrations: CalibrationStore,
    time_sync: TimeSync,
) -> PipelineManager:
    runners = []
    for pipeline_type in PIPELINE_TYPES:
        context = PipelineContext(
            inst, config.device_id, pipeline_type.NAME, calibrations, time_sync
        )
        enabled = config.pipelines_enabled.get(pipeline_type.NAME, False)
        runners.append(
            PipelineRunner(pipeline_type, context, enabled_by_default=enabled)
        )
    return PipelineManager(runners)


def with_simulated_calibrations(config: LocalConfig) -> LocalConfig:
    """Give cameras without a calibration file a plausible one, so the pipeline can run in sim."""
    width, height = config.camera_resolution_width, config.camera_resolution_height
    calibrations = []
    for i, calibration in enumerate(config.calibrations):
        if calibration is None:
            log.info(
                "Simulation: camera %d has no calibration; using a default 70 deg pinhole",
                i,
            )
            calibration = CameraCalibration(
                default_intrinsics(width, height), numpy.zeros((1, 5)), width, height
            )
        calibrations.append(calibration)
    return dataclasses.replace(config, calibrations=tuple(calibrations))


def run(
    config: LocalConfig, calibration_dir: Path, logs: LogBuffer, simulate: bool
) -> None:
    if simulate:
        config = with_simulated_calibrations(config)
    calibrations = CalibrationStore(calibration_dir, config.calibrations)
    inst, time_sync = connect_nt(config)
    status = StatusPublisher(inst, config.device_id, config.num_cameras)
    status.start()

    capture_settings = NTSettings(
        inst,
        f"/{config.device_id}/config",
        CaptureSettings,
        defaults=CaptureSettings(
            camera_id=config.camera_id,
            camera_resolution_width=config.camera_resolution_width,
            camera_resolution_height=config.camera_resolution_height,
            camera_exposure=config.camera_exposure,
            camera_gain=config.camera_gain,
        ),
    )
    capture: Capture
    if simulate:
        log.info(
            "SIMULATION MODE: rendering cameras from /%s/sim/true_pose",
            config.device_id,
        )
        capture = SimulatedCapture(
            inst, config.device_id, config.num_cameras, config.calibrations, time_sync
        )
    else:
        capture = MultiCameraCapture(config.num_cameras)

    pipelines = build_pipelines(config, inst, calibrations, time_sync)
    pipelines.start()

    streams = DebugStreams(config.num_cameras, pipelines.overlays)
    calibration = CalibrationSession(
        config.num_cameras, calibrations, calibration_dir / "calibration_board.json"
    )
    DashboardServer(
        config,
        inst,
        time_sync,
        status,
        capture,
        pipelines,
        streams,
        logs,
        calibrations,
        calibration,
    ).start()

    while True:
        settings = capture_settings.poll()
        if settings is not None:
            capture.apply_settings(settings)

        frames = capture.read()
        if frames is None:
            log.info("Waiting for camera config (camera_id, resolution)")
            time.sleep(0.5)
            continue
        status.frame_received(frames)
        pipelines.submit(frames)
        streams.set_frames(frames)
        calibration.submit(frames)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--calibration-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--sim",
        action="store_true",
        help="simulate the cameras (see src/sim/simulated_capture.py)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logs = LogBuffer()  # recent log lines for the dashboard
    logging.getLogger().addHandler(logs)
    run(
        load_local_config(args.config, args.calibration_dir),
        args.calibration_dir,
        logs,
        simulate=args.sim,
    )


if __name__ == "__main__":
    main()
