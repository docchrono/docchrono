from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from typing import overload

from docchrono.domain import Event, TemporalExpression

_ISO_PREFIX = re.compile(
    r"^(?P<year>[+-]?\d{4,})(?:-(?P<month>\d{2}))?(?:-(?P<day>\d{2}))?"
    r"(?:[T ](?P<hour>\d{2}))?(?::(?P<minute>\d{2}))?"
    r"(?::(?P<second>\d{2}))?(?:\.(?P<fraction>\d+))?"
)


class Timeline(Sequence[Event]):
    """A deterministic view of dated events with undated events kept explicitly.

    Iteration and indexing cover dated events only.  Callers must opt in to the
    separate :attr:`undated` collection, which prevents unresolved dates from
    being silently mixed into or dropped from the chronology.
    """

    __slots__ = ("_dated", "_undated")

    def __init__(self, events: Iterable[Event]) -> None:
        dated: list[Event] = []
        undated: list[Event] = []
        for event in events:
            if _resolved_temporals(event):
                dated.append(event)
            else:
                undated.append(event)
        self._dated = tuple(sorted(dated, key=_event_sort_key))
        self._undated = tuple(sorted(undated, key=_undated_sort_key))

    @property
    def dated(self) -> tuple[Event, ...]:
        return self._dated

    @property
    def undated(self) -> tuple[Event, ...]:
        return self._undated

    @property
    def all(self) -> tuple[Event, ...]:
        """Return dated events followed by the explicit undated collection."""

        return self._dated + self._undated

    def __len__(self) -> int:
        return len(self._dated)

    @overload
    def __getitem__(self, index: int) -> Event: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[Event, ...]: ...

    def __getitem__(self, index: int | slice) -> Event | tuple[Event, ...]:
        return self._dated[index]

    def __iter__(self) -> Iterator[Event]:
        return iter(self._dated)

    def __repr__(self) -> str:
        return f"Timeline(dated={len(self._dated)}, undated={len(self._undated)})"


def _resolved_temporals(event: Event) -> tuple[TemporalExpression, ...]:
    return tuple(
        temporal
        for temporal in event.temporal
        if temporal.resolved and (temporal.start is not None or temporal.end is not None)
    )


def _event_sort_key(event: Event) -> tuple[object, ...]:
    temporals = _resolved_temporals(event)
    bounds = tuple(_temporal_bounds(temporal) for temporal in temporals)
    lower = min(bound[0] for bound in bounds)
    upper = max(bound[1] for bound in bounds)
    competing_values = tuple(sorted(_temporal_identity(temporal) for temporal in temporals))
    return lower, upper, competing_values, event.title.casefold(), event.title, event.id


def _undated_sort_key(event: Event) -> tuple[object, ...]:
    values = tuple(sorted(_temporal_identity(temporal) for temporal in event.temporal))
    return event.title.casefold(), event.title, event.id, values


def _temporal_bounds(
    temporal: TemporalExpression,
) -> tuple[tuple[object, ...], tuple[object, ...]]:
    lower_text = temporal.start if temporal.start is not None else temporal.end
    upper_text = temporal.end if temporal.end is not None else temporal.start
    assert lower_text is not None
    assert upper_text is not None
    return _sortable_temporal(lower_text, upper=False), _sortable_temporal(upper_text, upper=True)


def _temporal_identity(temporal: TemporalExpression) -> tuple[object, ...]:
    return (
        temporal.start or "",
        temporal.end or "",
        temporal.precision.value,
        temporal.timezone or "",
        temporal.original_text,
        temporal.is_relative,
        temporal.resolved,
        temporal.evidence_span_id or "",
    )


def _sortable_temporal(value: str, *, upper: bool) -> tuple[object, ...]:
    """Turn an ISO-like boundary into a stable, timezone-independent sort key.

    Temporal normalization is performed upstream.  This function deliberately
    does not consult the host locale, timezone, or clock.  Partial ISO values use
    the beginning/end of their represented interval; unfamiliar values retain a
    deterministic lexical fallback instead of being guessed.
    """

    match = _ISO_PREFIX.match(value)
    if match is None:
        return (1, value.casefold(), value)

    parts = match.groupdict()
    month = int(parts["month"]) if parts["month"] else (12 if upper else 1)
    day = int(parts["day"]) if parts["day"] else (31 if upper else 1)
    hour = int(parts["hour"]) if parts["hour"] else (23 if upper else 0)
    minute = int(parts["minute"]) if parts["minute"] else (59 if upper else 0)
    second = int(parts["second"]) if parts["second"] else (59 if upper else 0)
    fraction_text = parts["fraction"] or ""
    fraction = int((fraction_text + "000000")[:6]) if fraction_text else (999999 if upper else 0)
    return (
        0,
        int(parts["year"]),
        month,
        day,
        hour,
        minute,
        second,
        fraction,
        value,
    )
