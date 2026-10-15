"""Bind a dataclass of settings to NetworkTables topics.

Every field of the dataclass becomes one topic in the given table:

    @dataclass(frozen=True)
    class Settings:
        fiducial_size_m: float = 0.1651    ->  <table>/fiducial_size_m (double)
        tag_layout: str = ""               ->  <table>/tag_layout      (string)

C2 sets each topic's default with ``setDefault``, so the topics show up and
can be edited in dashboards. A value published by the robot, or typed into a
dashboard, always wins over the default, even if C2 restarts later.
"""

from __future__ import annotations

import dataclasses
import typing
from typing import Any, Generic, TypeVar

import ntcore

S = TypeVar("S")

_TOPIC_GETTERS = {
    bool: "getBooleanTopic",
    int: "getIntegerTopic",
    float: "getDoubleTopic",
    str: "getStringTopic",
}


class NTSettings(Generic[S]):
    def __init__(
        self,
        inst: ntcore.NetworkTableInstance,
        table_path: str,
        settings_type: type[S],
        defaults: S | None = None,
    ) -> None:
        if not dataclasses.is_dataclass(settings_type):
            raise TypeError(f"{settings_type.__name__} must be a dataclass")
        self._type = settings_type
        defaults = defaults if defaults is not None else settings_type()
        hints = typing.get_type_hints(settings_type)
        table = inst.getTable(table_path)

        self._entries: dict[str, Any] = {}
        for field in dataclasses.fields(settings_type):
            field_type = hints[field.name]
            getter = _TOPIC_GETTERS.get(field_type)
            if getter is None:
                raise TypeError(
                    f"{settings_type.__name__}.{field.name}: unsupported type {field_type}; "
                    f"use one of {[t.__name__ for t in _TOPIC_GETTERS]}"
                )
            default = getattr(defaults, field.name)
            entry = getattr(table, getter)(field.name).getEntry(default)
            entry.setDefault(default)
            self._entries[field.name] = entry

        self._last_values: tuple | None = None
        self._current: S = defaults

    @property
    def current(self) -> S:
        return self._current

    def poll(self) -> S | None:
        """Read the topics. Returns new settings if anything changed since the last poll, else None."""
        values = tuple(entry.get() for entry in self._entries.values())
        if values == self._last_values:
            return None
        self._last_values = values
        self._current = self._type(**dict(zip(self._entries.keys(), values)))
        return self._current
