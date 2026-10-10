from datetime import datetime
from fractions import Fraction
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from time import monotonic_ns

import cv2


class RawFrameRecorder:
    """Record uncompressed pixels in MKV with capture-time timestamps.

    Every submitted frame is queued without a size limit. If encoding or
    disk writes fall behind, the backlog grows in memory until it is drained.
    The capture thread only copies the frame and performs a nonblocking enqueue.
    """

    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self._queue = Queue()
        self._stop = Event()
        self._failed = Event()
        self._thread = None
        self.path = None

    def __enter__(self):
        if self._enabled:
            # Local wall-clock time at recording startup, including UTC offset.
            timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f%z")
            self.path = Path(f"match_{timestamp}.mkv")
            self._thread = Thread(target=self._write_frames, name="raw-frame-recorder", daemon=True)
            self._thread.start()
        return self

    def submit(self, frame, timestamp_ns=None) -> None:
        if not self._enabled or self._failed.is_set() or self._stop.is_set():
            return
        # Own the pixels before detection/overlays can mutate the capture.
        if timestamp_ns is None:
            timestamp_ns = monotonic_ns()
        self._queue.put_nowait((timestamp_ns, frame.copy()))

    def _write_frames(self) -> None:
        try:
            # Load only when recording is enabled.
            import av

            print(f"Recording raw frames to {self.path}")
            time_base = Fraction(1, 1000)  # Matroska timestamps use milliseconds.
            stream = None
            first_timestamp = None
            last_pts = -1
            # Matroska stores uncompressed BGR via its VFW compatibility mode.
            with av.open(str(self.path), "w", format="matroska",
                         options={"allow_raw_vfw": "1"}) as video:
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        timestamp_ns, frame = self._queue.get(timeout=0.1)
                    except Empty:
                        continue
                    if len(frame.shape) == 2:
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    if stream is None:
                        stream = video.add_stream("rawvideo")
                        stream.width = frame.shape[1]
                        stream.height = frame.shape[0]
                        stream.pix_fmt = "bgr24"
                        stream.time_base = time_base
                        stream.codec_context.time_base = time_base
                        first_timestamp = timestamp_ns
                    if frame.shape != (stream.height, stream.width, 3):
                        raise ValueError("Capture resolution changed during recording")
                    raw_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
                    # Strictly increasing PTS also handles sub-millisecond captures.
                    raw_frame.pts = max(last_pts + 1, (timestamp_ns - first_timestamp) // 1_000_000)
                    raw_frame.time_base = time_base
                    last_pts = raw_frame.pts
                    # rawvideo packs pixels into packets; it performs no compression.
                    for packet in stream.encode(raw_frame):
                        video.mux(packet)
                if stream is not None:
                    for packet in stream.encode():
                        video.mux(packet)
        except Exception as exc:
            self._failed.set()
            print(f"Raw frame recording stopped: {exc}")
        finally:
            # Release queued frame memory if recording failed.
            while True:
                try:
                    self._queue.get_nowait()
                except Empty:
                    break

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            if not self._failed.is_set():
                print("Raw frame recording finished; all queued frames written")
