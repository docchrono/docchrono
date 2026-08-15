from __future__ import annotations

import math
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, cast


def freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a deeply immutable, JSON-compatible mapping."""

    frozen = _freeze_json(value)
    if not isinstance(frozen, Mapping):  # pragma: no cover - guarded by the type contract
        raise TypeError("expected a mapping")
    return cast("Mapping[str, Any]", frozen)


def thaw_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Convert frozen metadata to ordinary JSON containers for serialization."""

    return {key: _thaw_json(item) for key, item in value.items()}


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        items: list[tuple[str, object]] = []
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise TypeError("metadata keys must be strings")
            items.append((key, _freeze_json(item)))
        return MappingProxyType(dict(sorted(items)))
    if isinstance(value, (list, tuple)):
        sequence = cast("list[object] | tuple[object, ...]", value)
        return tuple(_freeze_json(item) for item in sequence)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("metadata numbers must be finite")
        return value
    raise TypeError(f"metadata value is not JSON-compatible: {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        return {str(key): _thaw_json(item) for key, item in mapping.items()}
    if isinstance(value, tuple):
        sequence = cast("tuple[object, ...]", value)
        return [_thaw_json(item) for item in sequence]
    return value
