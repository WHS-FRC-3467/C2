"""Keeps the most recent log lines in memory so the dashboard can show them."""

from __future__ import annotations

import logging
import threading
from collections import deque


class LogBuffer(logging.Handler):
    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self._lock_records = threading.Lock()
        self._records: deque[dict] = deque(maxlen=capacity)
        self._next_id = 1
        self.setFormatter(logging.Formatter("%(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:  # noqa: BLE001
            self.handleError(record)
            return
        with self._lock_records:
            self._records.append(
                {
                    "id": self._next_id,
                    "time": record.created,
                    "level": record.levelname,
                    "message": message,
                }
            )
            self._next_id += 1

    def since(self, after_id: int) -> list[dict]:
        """Records with id > ``after_id``, oldest first."""
        with self._lock_records:
            return [r for r in self._records if r["id"] > after_id]
