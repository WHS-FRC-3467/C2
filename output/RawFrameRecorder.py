from datetime import datetime
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Thread

import cv2


class RawFrameRecorder:
    """Record unmodified capture frames as a Motion JPEG stream.

    The queue holds at most eight frames. Overload drops new frames.
    The only work on the capture
    thread is a frame copy and a nonblocking enqueue.
    """

    def __init__(self, enabled: bool) -> None:
        self._enabled = enabled
        self._queue = Queue(maxsize=8)
        self._stop = Event()
        self._failed = Event()
        self._thread = None
        self.dropped_frames = 0
        self.path = None

    def __enter__(self):
        if self._enabled:
            # Local wall-clock time at recording startup, including UTC offset.
            timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f%z")
            self.path = Path(f"match_{timestamp}.mjpeg")
            self._thread = Thread(target=self._write_frames, name="raw-frame-recorder", daemon=True)
            self._thread.start()
        return self

    def submit(self, frame) -> None:
        if not self._enabled or self._failed.is_set() or self._stop.is_set():
            return
        if self._queue.full():
            self.dropped_frames += 1
            return
        try:
            # Own the pixels before detection/overlays can mutate the capture.
            self._queue.put_nowait(frame.copy())
        except Full:
            self.dropped_frames += 1

    def _write_frames(self) -> None:
        try:
            print(f"Recording raw frames to {self.path}")
            # Concatenated JPEG images form a raw MJPEG stream, which does
            # not require a fixed capture rate or a video container index.
            with self.path.open("wb") as video:
                while not self._stop.is_set() or not self._queue.empty():
                    try:
                        frame = self._queue.get(timeout=0.1)
                    except Empty:
                        continue
                    success, encoded = cv2.imencode(
                        ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if not success:
                        raise OSError("Could not encode raw frame as JPEG")
                    video.write(memoryview(encoded))
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
            print(f"Raw frame recording finished; dropped {self.dropped_frames} frames")
