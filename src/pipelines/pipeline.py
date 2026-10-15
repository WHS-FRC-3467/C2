"""What a pipeline is, and what it gets to work with.

A pipeline is a class that:

* reads frames from the multi-camera module, flatbuffer messages from NT, or both,
* writes flatbuffer messages to NT through ``context.output(...)``,
* is enabled, disabled, and configured from NT (see pipelines/runner.py for topics).

Minimal example::

    @dataclass(frozen=True)
    class MySettings:
        threshold: float = 0.5           # -> /<dev>/pipelines/my_pipeline/config/threshold

    class MyPipeline(Pipeline[MySettings]):
        NAME = "my_pipeline"
        SETTINGS = MySettings
        NT_INPUTS = {"robot_pose": NTInput("/robot/pose", "robotpose_fb")}

        def __init__(self, context):
            super().__init__(context)
            self._out = context.output("result", "mything_fb")

        def configure(self, settings):
            self._threshold = settings.threshold

        def process(self, inputs):
            # Robot-clock timestamp of the frame (None while not synced with
            # the robot, in which case publish() drops the message).
            stamp = self.context.time_sync.stamp(inputs.frames.capture_time_us)
            self._out.publish(build_flatbuffer(...), stamp)
            return None                  # or an Overlay for the debug stream

Register the class in pipelines/__init__.py.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar, Generic, TypeVar

import ntcore
import numpy

from config.local_config import CameraCalibration
from core.frames import FrameSet
from core.nt_io import FlatbufferMessage, FlatbufferPublisher
from core.time_sync import TimeSync

S = TypeVar("S")

Overlay = Callable[[Sequence[numpy.ndarray]], None]
"""Draws a pipeline's debug overlay onto per-camera BGR images (for the MJPEG stream)."""


@dataclass(frozen=True)
class NTInput:
    """A flatbuffer topic a pipeline wants to read. ``{device_id}`` in the topic is filled in."""

    topic: str
    type_string: str
    queue_depth: int = 32


@dataclass(frozen=True)
class PipelineInputs:
    frames: FrameSet | None
    """The newest frame set (None for pipelines with USES_CAMERAS = False).
    Frames that arrived while the pipeline was busy are skipped; check
    ``frames.sequence`` if you need to know."""

    messages: dict[str, list[FlatbufferMessage]]
    """Every message received on each NT_INPUTS topic since the last call, oldest first."""


class PipelineContext:
    """Everything a pipeline may touch outside itself."""

    def __init__(
        self,
        inst: ntcore.NetworkTableInstance,
        device_id: str,
        pipeline_name: str,
        calibrations: Sequence[CameraCalibration | None],
        time_sync: TimeSync,
    ) -> None:
        self.inst = inst
        self.device_id = device_id
        self.calibrations = tuple(calibrations)
        self.time_sync = time_sync
        self.table_path = f"/{device_id}/pipelines/{pipeline_name}"
        self._outputs: dict[str, FlatbufferPublisher] = {}

    @property
    def num_cameras(self) -> int:
        return len(self.calibrations)

    def output_topics(self) -> list[str]:
        return [publisher.topic for publisher in self._outputs.values()]

    def output(self, name: str, type_string: str) -> FlatbufferPublisher:
        """The publisher for ``/<dev>/pipelines/<pipeline>/output/<name>``.

        Publishers outlive pipeline instances (a pipeline is rebuilt each time
        it is enabled), so subscribers on the robot never see the topic vanish.
        """
        if name not in self._outputs:
            self._outputs[name] = FlatbufferPublisher(self.inst, f"{self.table_path}/output/{name}", type_string)
        return self._outputs[name]


class Pipeline(ABC, Generic[S]):
    NAME: ClassVar[str]
    """Unique name. Also the NT table: /<device_id>/pipelines/<NAME>/"""

    SETTINGS: ClassVar[type]
    """Frozen dataclass of NT-configurable settings (bool/int/float/str fields with defaults)."""

    USES_CAMERAS: ClassVar[bool] = True
    """If True, process() runs once per new frame set. If False, it runs
    whenever an NT input has new messages."""

    NT_INPUTS: ClassVar[dict[str, NTInput]] = {}

    def __init__(self, context: PipelineContext) -> None:
        """Called each time the pipeline is enabled. Load models and create outputs here."""
        self.context = context

    def configure(self, settings: S) -> None:  # noqa: B027 - optional hook
        """Called with the current settings right after __init__, and again whenever they change."""

    @abstractmethod
    def process(self, inputs: PipelineInputs) -> Overlay | None:
        """Do the work and publish results. Optionally return a debug overlay."""

    def summary(self) -> dict | list | None:
        """Optional: the latest results as plain JSON-able data, shown on the dashboard.

        Called from the web server's thread, so return a snapshot (e.g. build it
        in process() and store it), don't compute from mutable state here.
        """
        return None

    def close(self) -> None:  # noqa: B027 - optional hook
        """Called when the pipeline is disabled. Release threads, GPU memory, etc."""
