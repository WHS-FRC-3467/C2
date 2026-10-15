"""Timestamped flatbuffer messages over NetworkTables.

Pipelines never call ntcore publishers directly. They use these wrappers, so
every message leaving C2 is timestamped the same way (see core/time_sync.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import ntcore

from core.time_sync import Timestamp, TimeSync

# Every message matters (the robot replays each one into its pose estimator),
# so queue them all instead of collapsing to the latest value. Send them
# within 5 ms instead of on ntcore's default 100 ms batch.
_MESSAGE_OPTIONS = dict(sendAll=True, keepDuplicates=True)
_PUBLISH_PERIOD_S = 0.005


class FlatbufferPublisher:
    """Publishes flatbuffer payloads on a raw topic, stamped with their capture time."""

    def __init__(
        self, inst: ntcore.NetworkTableInstance, topic: str, type_string: str
    ) -> None:
        self.topic = topic
        self._pub = inst.getRawTopic(topic).publish(
            type_string,
            ntcore.PubSubOptions(periodic=_PUBLISH_PERIOD_S, **_MESSAGE_OPTIONS),
        )

    def publish(self, payload: bytes, timestamp: Timestamp | None) -> bool:
        """Publish ``payload`` as if it was produced at ``timestamp``.

        On the robot, ``TimestampedRaw.timestamp`` for this value will equal
        ``timestamp.server_us``. Pass ``TimeSync.stamp(...)`` straight in: if
        it is None (not synced with the robot), the message is dropped,
        because it could not be timestamped correctly. Returns whether it was
        published.
        """
        if timestamp is None:
            return False
        self._pub.set(payload, timestamp.publish_us)
        return True


@dataclass(frozen=True)
class FlatbufferMessage:
    payload: bytes
    time_us: int
    """When the sender says the data is from, in this coprocessor's NT time base
    (comparable to ``FrameSet.capture_time_us`` and ``nt_now_us()``)."""


class FlatbufferSubscriber:
    """Receives every flatbuffer message published on a raw topic since the last read."""

    def __init__(
        self,
        inst: ntcore.NetworkTableInstance,
        topic: str,
        type_string: str,
        time_sync: TimeSync,
        queue_depth: int = 32,
    ) -> None:
        self.topic = topic
        self._time_sync = time_sync
        self._sub = inst.getRawTopic(topic).subscribe(
            type_string,
            b"",
            ntcore.PubSubOptions(pollStorage=queue_depth, **_MESSAGE_OPTIONS),
        )

    def read(self) -> list[FlatbufferMessage]:
        return [
            FlatbufferMessage(
                value.value, self._time_sync.received_to_local_us(value.time)
            )
            for value in self._sub.readQueue()
            if value.value
        ]
