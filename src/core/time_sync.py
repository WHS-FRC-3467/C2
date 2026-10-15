"""Timestamps for everything C2 sends to (and receives from) NetworkTables.

There are three clocks in play:

* CLOCK_MONOTONIC on this coprocessor. V4L2 stamps every camera buffer with it.
* The coprocessor's NT clock, ``ntcore._now()``. This is ``wpi::Now()``: the
  wall-clock time at process start plus elapsed CLOCK_MONOTONIC time, in us.
  It is NOT ``time.time()``. The two diverge after any NTP/clock step.
* The robot's NT clock (the NT server). On a roboRIO this is the FPGA
  timestamp, the same clock as ``Timer.getFPGATimestamp()`` and
  ``WPIUtilJNI.now()``.

ntcore (verified against allwpilib v2026.2.1, net/ClientImpl.cpp and
net/NetworkOutgoingQueue.h) converts between the last two for us:

* When the client connects it sends ONE ping. From the reply it computes
  ``offset = server_time - local_time`` (``getServerTimeOffset()``). Nothing
  is sent to the server until that offset exists.
* Every outgoing value timestamp gets ``+ offset`` added on the wire, and
  every incoming one gets ``- offset``. So a value published with a
  timestamp in ``ntcore._now()`` time shows up on the robot in FPGA time.

The catch: the offset is measured once per connection and never updated.
Because the coprocessor and roboRIO oscillators drift (typically tens of
ppm), the error grows for as long as the connection stays up. For example,
30 ppm over a 20 minute connection is 36 ms. The single measurement is also
taken during the connection handshake, when latency is at its worst.

``TimeSync`` fixes both problems. Every few seconds it opens a short-lived
second NT client (a "probe"), lets ntcore measure a fresh offset on it, and
keeps the lowest-latency recent measurement. Before publishing, it shifts the
timestamp by ``fresh_offset - connection_offset``, so that after ntcore adds
its stale offset the robot receives exactly ``local_time + fresh_offset``.
"""

from __future__ import annotations

import atexit
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass

import ntcore

log = logging.getLogger(__name__)


def nt_now_us() -> int:
    """Current time in the coprocessor's NT time base, in microseconds."""
    return ntcore._now()


def monotonic_to_nt_us(monotonic_s: float) -> int:
    """Convert a CLOCK_MONOTONIC time (seconds) to the coprocessor's NT time base.

    Both clocks run at the same rate (wpi::Now is driven by steady_clock), so
    the age of the event is the same on both clocks.
    """
    age_us = round((time.monotonic() - monotonic_s) * 1_000_000)
    return nt_now_us() - age_us


@dataclass(frozen=True)
class Timestamp:
    """One instant, expressed in both time bases, computed once so they agree.

    ``server_us``: the robot's NT clock (FPGA time on a roboRIO). This is the
        timestamp the robot will see on the published value.
    ``publish_us``: the value to hand to an ntcore publisher's ``set()``.
        ntcore adds its connection offset to this on the wire, which lands it
        exactly on ``server_us``.
    """

    local_us: int
    server_us: int
    publish_us: int


@dataclass(frozen=True)
class _OffsetSample:
    offset_us: int
    rtt2_us: int
    measured_at_s: float
    # ntcore's offset for the main connection when this sample was taken. A
    # sample is only trusted while that is unchanged; a new value means a new
    # connection, possibly to a rebooted robot whose clock started over.
    connection_offset_us: int


class TimeSync:
    """Tracks the offset between this coprocessor's NT clock and the robot's.

    It is thread-safe. Call ``start()`` once after the NT client is running.
    """

    def __init__(
        self,
        inst: ntcore.NetworkTableInstance,
        server_address: str,
        client_name: str,
        probe_interval_s: float = 10.0,
        server_port: int = ntcore.NetworkTableInstance.kDefaultPort4,
        probe_window: int = 3,
        diagnostics_table: str | None = None,
    ) -> None:
        self._inst = inst
        self._server_address = server_address
        self._server_port = server_port
        self._client_name = client_name
        self._probe_interval_s = probe_interval_s
        self._samples: deque[_OffsetSample] = deque(maxlen=max(1, probe_window))
        self._lock = threading.Lock()
        self._wake = threading.Event()

        self._diag = None
        if diagnostics_table is not None:
            table = inst.getTable(diagnostics_table)
            self._diag = {
                name: table.getIntegerTopic(name).publish()
                for name in (
                    "connection_offset_us",
                    "probe_offset_us",
                    "probe_rtt2_us",
                    "correction_us",
                )
            }

    # ---------------------------------------------------------------- public

    def start(self) -> None:
        listener = self._inst.addTimeSyncListener(False, self._on_connection_time_sync)
        # pyntcore (2026.2.2) aborts the interpreter at exit ("terminate called
        # without an active exception") if a time sync listener is still
        # registered, which would turn sys.exit(1) restarts into core dumps.
        atexit.register(self._inst.removeListener, listener)
        if self._probe_interval_s > 0:
            threading.Thread(
                target=self._probe_loop, name="time-sync-probe", daemon=True
            ).start()

    def snapshot(self) -> dict:
        """Current sync state, for the dashboard. Offsets are robot clock minus ours, in us."""
        connection = self._inst.getServerTimeOffset()
        best = None
        if connection is not None:
            with self._lock:
                best = self._best_sample(connection)
        return {
            "synced": connection is not None,
            "connection_offset_us": connection,
            "probe_offset_us": best.offset_us if best else None,
            "probe_rtt2_us": best.rtt2_us if best else None,
            "probe_age_s": time.monotonic() - best.measured_at_s if best else None,
            "correction_us": (
                best.offset_us - connection if best and connection is not None else None
            ),
            "probe_interval_s": self._probe_interval_s,
        }

    def server_offset_us(self) -> int | None:
        """Best estimate of ``server_time - local_time``, or None if not connected."""
        offsets = self._offsets()
        return None if offsets is None else offsets[1]

    def stamp(self, local_us: int) -> Timestamp | None:
        """Express a local NT-clock time on the robot's clock.

        Returns None while there is no connection or no offset yet, because
        nothing published then could be timestamped correctly. Callers should
        drop the data in that case.
        """
        offsets = self._offsets()
        if offsets is None:
            return None
        connection, best = offsets
        # ntcore treats a timestamp of 0 as "now", so never hand it 0.
        publish_us = max(1, local_us + best - connection)
        return Timestamp(
            local_us=local_us, server_us=local_us + best, publish_us=publish_us
        )

    def received_to_local_us(self, nt_time_us: int) -> int:
        """Correct the timestamp of a value received from NT (``TimestampedX.time``).

        ntcore already subtracted the stale connection offset. This swaps it
        for the fresh one, so the result is comparable with ``nt_now_us()``
        and with capture timestamps.
        """
        offsets = self._offsets()
        if offsets is None:
            return nt_time_us
        connection, best = offsets
        return nt_time_us + connection - best

    # --------------------------------------------------------------- private

    def _offsets(self) -> tuple[int, int] | None:
        """Return (ntcore's connection offset, best offset estimate), or None if not synced."""
        # Always ask ntcore. It is the offset ntcore will actually apply on
        # the wire, and it returns 0 when we are the server.
        connection = self._inst.getServerTimeOffset()
        if connection is None:
            return None
        with self._lock:
            best = self._best_sample(connection)
        return connection, (best.offset_us if best is not None else connection)

    def _best_sample(self, connection_offset_us: int) -> _OffsetSample | None:
        """Lowest-latency recent sample from the current connection. Caller holds the lock."""
        max_age_s = self._probe_interval_s * (self._samples.maxlen or 1) * 1.5
        now = time.monotonic()
        usable = [
            s
            for s in self._samples
            if s.connection_offset_us == connection_offset_us
            and now - s.measured_at_s <= max_age_s
        ]
        return min(usable, key=lambda s: s.rtt2_us) if usable else None

    def _is_server(self) -> bool:
        return bool(
            self._inst.getNetworkMode()
            & ntcore.NetworkTableInstance.NetworkMode.kNetModeServer
        )

    def _on_connection_time_sync(self, event: ntcore.Event) -> None:
        data = event.data
        if data.valid:
            log.info(
                "NT time synced: offset=%d us, rtt/2=%d us",
                data.serverTimeOffset,
                data.rtt2,
            )
            self._wake.set()  # probe now instead of waiting a full interval
        else:
            log.warning("NT time sync lost (disconnected)")

    def _probe_loop(self) -> None:
        while True:
            self._wake.wait(timeout=self._probe_interval_s)
            self._wake.clear()
            connection = self._inst.getServerTimeOffset()
            if connection is None or self._is_server():
                continue
            try:
                sample = self._probe_once(connection)
            except Exception:  # noqa: BLE001 - never let the probe thread die
                log.exception("Time sync probe failed")
                continue
            if sample is None:
                log.warning("Time sync probe got no response")
                continue
            with self._lock:
                self._samples.append(sample)
                best = self._best_sample(connection)
            log.debug(
                "Time sync probe: offset=%d us rtt/2=%d us",
                sample.offset_us,
                sample.rtt2_us,
            )
            if self._diag is not None and best is not None:
                self._diag["connection_offset_us"].set(connection)
                self._diag["probe_offset_us"].set(best.offset_us)
                self._diag["probe_rtt2_us"].set(best.rtt2_us)
                self._diag["correction_us"].set(best.offset_us - connection)

    def _probe_once(
        self, connection_offset_us: int, timeout_s: float = 2.0
    ) -> _OffsetSample | None:
        """Connect a throwaway NT client and let ntcore measure a fresh offset."""
        probe = ntcore.NetworkTableInstance.create()
        result: list[_OffsetSample] = []
        done = threading.Event()

        def on_sync(event: ntcore.Event) -> None:
            if event.data.valid and not done.is_set():
                result.append(
                    _OffsetSample(
                        offset_us=event.data.serverTimeOffset,
                        rtt2_us=event.data.rtt2,
                        measured_at_s=time.monotonic(),
                        connection_offset_us=connection_offset_us,
                    )
                )
                done.set()

        # Its connect/disconnect every interval is expected; only log problems.
        probe.configPythonLogging(
            min=ntcore.NetworkTableInstance.LogLevel.kLogWarning,
            name="nt.timesync_probe",
        )
        try:
            probe.addTimeSyncListener(False, on_sync)
            probe.setServer(self._server_address, self._server_port)
            probe.startClient4(self._client_name)
            done.wait(timeout_s)
        finally:
            probe.stopClient()
            ntcore.NetworkTableInstance.destroy(probe)
        return result[0] if result else None
