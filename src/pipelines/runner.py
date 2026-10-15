"""Runs each pipeline on its own thread and wires it to NetworkTables.

NT layout for a pipeline named <name> on device <dev>:

    /<dev>/pipelines/<name>/enabled           bool    turn the pipeline on/off
    /<dev>/pipelines/<name>/config/<field>    any     one topic per SETTINGS field
    /<dev>/pipelines/<name>/output/<output>   raw     flatbuffer results
    /<dev>/pipelines/<name>/status/active     bool    enabled and running without errors
    /<dev>/pipelines/<name>/status/fps        double  process() calls per second
    /<dev>/pipelines/<name>/status/latency_ms double  capture -> results published
    /<dev>/pipelines/<name>/status/error      string  last exception, "" if none

``enabled`` and ``config/*`` get defaults via setDefault, so the robot (or a
dashboard) can override them at any time.
"""

from __future__ import annotations

import logging
import threading
import time

from core.frames import FrameSet
from core.nt_io import FlatbufferSubscriber
from core.nt_settings import NTSettings
from core.time_sync import nt_now_us
from pipelines.pipeline import Overlay, Pipeline, PipelineContext, PipelineInputs

log = logging.getLogger(__name__)

_IDLE_POLL_S = 0.1
_NT_INPUT_POLL_S = 0.005
_STATUS_PERIOD_S = 1.0
_START_RETRY_S = 5.0


class PipelineRunner:
    def __init__(self, pipeline_type: type[Pipeline], context: PipelineContext, enabled_by_default: bool) -> None:
        self.name = pipeline_type.NAME
        self._type = pipeline_type
        self._context = context
        table = context.inst.getTable(context.table_path)

        self._enabled = table.getBooleanTopic("enabled").getEntry(enabled_by_default)
        self._enabled.setDefault(enabled_by_default)
        self._settings = NTSettings(context.inst, f"{context.table_path}/config", pipeline_type.SETTINGS)
        self._inputs = {
            key: FlatbufferSubscriber(
                context.inst,
                spec.topic.format(device_id=context.device_id),
                spec.type_string,
                context.time_sync,
                spec.queue_depth,
            )
            for key, spec in pipeline_type.NT_INPUTS.items()
        }

        status = table.getSubTable("status")
        self._active_pub = status.getBooleanTopic("active").publish()
        self._fps_pub = status.getDoubleTopic("fps").publish()
        self._latency_pub = status.getDoubleTopic("latency_ms").publish()
        self._error_pub = status.getStringTopic("error").publish()
        self._active_pub.set(False)
        self._error_pub.set("")

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._frame_lock = threading.Condition()
        self._pending_frames: FrameSet | None = None
        self._overlay: Overlay | None = None
        self._pipeline: Pipeline | None = None
        self._next_start_attempt = 0.0

    # -------------------------------------------------- called by other threads

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=f"pipeline-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the thread and close the pipeline (used by tests and shutdown)."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join()

    def submit(self, frames: FrameSet) -> None:
        """Hand over the newest frames. An unprocessed older frame set is dropped."""
        if not self._type.USES_CAMERAS:
            return
        with self._frame_lock:
            self._pending_frames = frames
            self._frame_lock.notify()

    @property
    def overlay(self) -> Overlay | None:
        return self._overlay if self._pipeline is not None else None

    # ------------------------------------------------------------ worker thread

    def _run(self) -> None:
        calls = 0
        last_status = time.monotonic()
        while not self._stop.is_set():
            if not self._update_enabled():
                time.sleep(_IDLE_POLL_S)
                continue
            assert self._pipeline is not None

            inputs = self._next_inputs()
            if inputs is None:
                continue

            try:
                settings = self._settings.poll()
                if settings is not None:
                    self._pipeline.configure(settings)
                self._overlay = self._pipeline.process(inputs)
                self._error_pub.set("")
                self._active_pub.set(True)
            except Exception as e:  # noqa: BLE001 - one bad frame must not kill the pipeline
                log.exception("Pipeline %s failed", self.name)
                self._error_pub.set(f"{type(e).__name__}: {e}")
                self._active_pub.set(False)
                time.sleep(_IDLE_POLL_S)  # don't spin on a persistent error
                continue

            calls += 1
            if inputs.frames is not None:
                self._latency_pub.set((nt_now_us() - inputs.frames.capture_time_us) / 1000.0)
            now = time.monotonic()
            if now - last_status >= _STATUS_PERIOD_S:
                self._fps_pub.set(calls / (now - last_status))
                calls, last_status = 0, now
        self._update_enabled(force_disable=True)

    def _update_enabled(self, force_disable: bool = False) -> bool:
        """Create or tear down the pipeline when ``enabled`` changes. Returns whether it is enabled."""
        enabled = self._enabled.get() and not force_disable
        if enabled and self._pipeline is None and time.monotonic() >= self._next_start_attempt:
            log.info("Enabling pipeline %s", self.name)
            try:
                self._pipeline = self._type(self._context)
                self._settings.poll()
                self._pipeline.configure(self._settings.current)
            except Exception as e:  # noqa: BLE001
                log.exception("Pipeline %s failed to start; retrying in %.0f s", self.name, _START_RETRY_S)
                self._error_pub.set(f"{type(e).__name__}: {e}")
                self._pipeline = None
                self._next_start_attempt = time.monotonic() + _START_RETRY_S
                return False
            for sub in self._inputs.values():
                sub.read()  # discard anything queued while we were disabled
        elif not enabled and self._pipeline is not None:
            log.info("Disabling pipeline %s", self.name)
            pipeline, self._pipeline = self._pipeline, None
            self._active_pub.set(False)
            try:
                pipeline.close()
            except Exception:  # noqa: BLE001
                log.exception("Pipeline %s failed to close", self.name)
        return self._pipeline is not None

    def _next_inputs(self) -> PipelineInputs | None:
        frames = None
        if self._type.USES_CAMERAS:
            with self._frame_lock:
                if self._pending_frames is None:
                    self._frame_lock.wait(_IDLE_POLL_S)
                frames, self._pending_frames = self._pending_frames, None
            if frames is None:
                return None

        messages = {key: sub.read() for key, sub in self._inputs.items()}
        if not self._type.USES_CAMERAS and not any(messages.values()):
            time.sleep(_NT_INPUT_POLL_S)
            return None
        return PipelineInputs(frames, messages)


class PipelineManager:
    """Creates one runner per registered pipeline type and fans frames out to them."""

    def __init__(self, runners: list[PipelineRunner]) -> None:
        names = [r.name for r in runners]
        if len(set(names)) != len(names):
            raise ValueError(f"Duplicate pipeline names: {names}")
        self.runners = runners

    def start(self) -> None:
        for runner in self.runners:
            runner.start()

    def stop(self) -> None:
        for runner in self.runners:
            runner.stop()

    def submit(self, frames: FrameSet) -> None:
        for runner in self.runners:
            runner.submit(frames)

    def overlays(self) -> list[Overlay]:
        return [o for o in (r.overlay for r in self.runners) if o is not None]
