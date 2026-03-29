from typing import Optional

import ntcore


def _nt_now_us() -> int:
    now_fn = getattr(ntcore, "_now", None)
    if now_fn is None:
        raise RuntimeError("ntcore does not expose a current-time function")
    return int(now_fn())


class NtServerClock:
    def __init__(
            self,
            nt_instance: Optional[ntcore.NetworkTableInstance] = None) -> None:
        self._nt_instance = nt_instance or ntcore.NetworkTableInstance.getDefault()
        self._server_time_offset_us: Optional[int] = None

    def now_us(self) -> Optional[int]:
        server_time_offset_us = self._nt_instance.getServerTimeOffset()
        if server_time_offset_us is not None:
            self._server_time_offset_us = int(server_time_offset_us)

        if self._server_time_offset_us is None:
            return None

        return _nt_now_us() + self._server_time_offset_us
