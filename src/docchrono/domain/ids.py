from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import TypeAlias, cast

from pydantic import BaseModel

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]


def content_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def stable_id(prefix: str, *parts: object) -> str:
    payload = json.dumps(
        [_canonical(part) for part in parts],
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:24]
    return f"{prefix}_{digest}"


def _canonical(value: object) -> JsonValue:
    if isinstance(value, Enum):
        return _canonical(value.value)
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, BaseModel):
        return _canonical(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        return {
            str(key): _canonical(item)
            for key, item in sorted(mapping.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        sequence = cast("Sequence[object]", value)
        return [_canonical(item) for item in sequence]
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
