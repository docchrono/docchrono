from docchrono.domain import Event, EventType, TemporalExpression, TemporalPrecision
from docchrono.views import Timeline


def _event(
    identifier: str,
    *temporals: TemporalExpression,
    title: str | None = None,
) -> Event:
    return Event(
        id=identifier,
        type=EventType.ACTION,
        title=title or identifier,
        claim_ids=(f"claim-{identifier}",),
        temporal=temporals,
        score=0.9,
    )


def _temporal(
    text: str,
    start: str | None,
    end: str | None = None,
    *,
    resolved: bool = True,
) -> TemporalExpression:
    return TemporalExpression(
        original_text=text,
        start=start,
        end=end,
        precision=TemporalPrecision.RANGE if end else TemporalPrecision.DAY,
        resolved=resolved,
    )


def test_orders_by_temporal_envelope_and_preserves_competing_values() -> None:
    competing = _event(
        "event-competing",
        _temporal("one source", "2022-05-01"),
        _temporal("another source", "2025-06-01"),
    )
    interval = _event(
        "event-interval",
        _temporal("during 2023", "2023-01-01", "2023-12-31"),
    )
    exact = _event("event-exact", _temporal("2024-01-01", "2024-01-01"))

    timeline = Timeline((exact, interval, competing))

    assert tuple(event.id for event in timeline) == (
        "event-competing",
        "event-interval",
        "event-exact",
    )
    assert timeline[0].temporal == competing.temporal
    assert len(timeline[0].temporal) == 2


def test_keeps_unresolved_and_missing_dates_explicitly_undated() -> None:
    dated = _event("dated", _temporal("2024", "2024"))
    unresolved = _event("a-unresolved", _temporal("next week", None, resolved=False))
    missing = _event("b-missing")

    timeline = Timeline((missing, dated, unresolved))

    assert timeline.dated == (dated,)
    assert tuple(event.id for event in timeline.undated) == ("a-unresolved", "b-missing")
    assert timeline.all == (dated, unresolved, missing)
