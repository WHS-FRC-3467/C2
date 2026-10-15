"""Device-wide health on NT, so the robot can tell C2 is alive and which cameras work.

    /<dev>/status/heartbeat            int     +1 every 100 ms while the process runs
    /<dev>/status/capture_fps          double  combined frames per second
    /<dev>/status/camera_<i>/connected bool    camera i is delivering images

The heartbeat comes from its own thread, so it keeps ticking even while the
capture loop is blocked waiting for the camera. A stalled camera shows up in
``connected`` instead of stopping the heartbeat.

All cameras arrive in one combined V4L2 frame, so per-camera health is
inferred: camera i is connected if a frame arrived recently AND its part of
the frame isn't perfectly uniform. A dead or unplugged sensor in the module
outputs a flat image. A covered or dark but working sensor still has noise.
"""

from __future__ import annotations

import logging
import threading
import time

import ntcore
import numpy

from core.frames import FrameSet

log = logging.getLogger(__name__)

HEARTBEAT_PERIOD_S = 0.1
FRAME_TIMEOUT_S = 0.5
_UNIFORMITY_STRIDE = 8  # check every 8th pixel each way; plenty to see sensor noise


class StatusPublisher:
    def __init__(self, inst: ntcore.NetworkTableInstance, device_id: str, num_cameras: int) -> None:
        table = inst.getTable(f"/{device_id}/status")
        self._heartbeat_pub = table.getIntegerTopic("heartbeat").publish()
        self._fps_pub = table.getDoubleTopic("capture_fps").publish()
        self._connected_pubs = [
            table.getSubTable(f"camera_{i}").getBooleanTopic("connected").publish() for i in range(num_cameras)
        ]

        self._lock = threading.Lock()
        self._last_frame_s = -float("inf")
        self._has_image = [False] * num_cameras
        self._frame_count = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="status", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def frame_received(self, frames: FrameSet) -> None:
        """Call from the capture loop for every frame set."""
        has_image = [_has_image(camera) for camera in frames.cameras]
        with self._lock:
            self._last_frame_s = time.monotonic()
            self._has_image = has_image
            self._frame_count += 1

    def _run(self) -> None:
        beat = 0
        last_frame_count = 0
        second_start_s = time.monotonic()
        while not self._stop.wait(HEARTBEAT_PERIOD_S):
            beat += 1
            self._heartbeat_pub.set(beat)
            now = time.monotonic()
            with self._lock:
                streaming = now - self._last_frame_s <= FRAME_TIMEOUT_S
                connected = [streaming and ok for ok in self._has_image]
                frame_count = self._frame_count
            for pub, ok in zip(self._connected_pubs, connected):
                pub.set(ok)

            if now - second_start_s >= 1.0:
                fps = (frame_count - last_frame_count) / (now - second_start_s)
                self._fps_pub.set(fps)
                log.info("Capture %.1f fps, cameras connected: %s", fps, connected)
                last_frame_count, second_start_s = frame_count, now


def _has_image(camera: numpy.ndarray) -> bool:
    sample = camera[::_UNIFORMITY_STRIDE, ::_UNIFORMITY_STRIDE]
    return bool(sample.size) and int(sample.max()) != int(sample.min())
