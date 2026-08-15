from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Generic, TypeVar, cast, overload

from docchrono.domain import CaseData, ReviewAction, ReviewDecision, ReviewItem
from docchrono.errors import ReviewDecisionError
from docchrono.review.decisions import create_decision
from docchrono.review.engine import ReviewEngine

SnapshotT = TypeVar("SnapshotT")


class ReviewBook(Generic[SnapshotT]):
    """The small interface for inspecting and applying auditable review decisions."""

    __slots__ = ("_case_factory", "_data", "_engine")

    @overload
    def __init__(
        self: ReviewBook[CaseData],
        data: CaseData,
        *,
        engine: ReviewEngine | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        data: CaseData,
        *,
        engine: ReviewEngine | None = None,
        case_factory: Callable[[CaseData], SnapshotT],
    ) -> None: ...

    def __init__(
        self,
        data: CaseData,
        *,
        engine: ReviewEngine | None = None,
        case_factory: Callable[[CaseData], SnapshotT] | None = None,
    ) -> None:
        self._data = data
        self._engine = engine or ReviewEngine()
        self._case_factory = case_factory or cast("Callable[[CaseData], SnapshotT]", _identity)

    @property
    def decisions(self) -> tuple[ReviewDecision, ...]:
        return self._data.review_decisions

    @property
    def pending(self) -> tuple[ReviewItem, ...]:
        effective = self._engine.effective_decisions(self._data.review_decisions)
        resolved_item_ids = {
            target_id
            for decision in effective
            if decision.action in {ReviewAction.ACCEPT, ReviewAction.REJECT}
            for target_id in decision.target_ids
        }
        return tuple(
            sorted(
                (item for item in self._data.review_items if item.id not in resolved_item_ids),
                key=lambda item: item.id,
            )
        )

    def accept(
        self,
        target_id: str,
        *,
        reason: str | None = None,
        supersedes: str | None = None,
    ) -> ReviewDecision:
        return create_decision(
            ReviewAction.ACCEPT,
            (target_id,),
            reason=reason,
            supersedes=supersedes,
        )

    def reject(
        self,
        target_id: str,
        *,
        reason: str | None = None,
        supersedes: str | None = None,
    ) -> ReviewDecision:
        return create_decision(
            ReviewAction.REJECT,
            (target_id,),
            reason=reason,
            supersedes=supersedes,
        )

    def merge_entities(
        self,
        entity_ids: Iterable[str],
        *,
        canonical_name: str | None = None,
        reason: str | None = None,
        supersedes: str | None = None,
    ) -> ReviewDecision:
        targets = tuple(sorted(set(entity_ids)))
        if len(targets) < 2:
            raise ReviewDecisionError("merging entities requires at least two entity ids")
        before = self._entity_merge_snapshot(targets)
        return create_decision(
            ReviewAction.MERGE_ENTITIES,
            targets,
            payload={"canonical_name": canonical_name, "before": before},
            reason=reason,
            supersedes=supersedes,
        )

    def unmerge_entity(
        self,
        entity_id: str,
        *,
        supersedes: str,
        reason: str | None = None,
    ) -> ReviewDecision:
        return create_decision(
            ReviewAction.UNMERGE_ENTITY,
            (entity_id,),
            reason=reason,
            supersedes=supersedes,
        )

    def set_canonical_name(
        self,
        entity_id: str,
        canonical_name: str,
        *,
        reason: str | None = None,
        supersedes: str | None = None,
    ) -> ReviewDecision:
        if not canonical_name.strip():
            raise ReviewDecisionError("canonical name must not be blank")
        return create_decision(
            ReviewAction.SET_CANONICAL_NAME,
            (entity_id,),
            payload={"canonical_name": canonical_name},
            reason=reason,
            supersedes=supersedes,
        )

    def group_events(
        self,
        event_ids: Iterable[str],
        *,
        reason: str | None = None,
        supersedes: str | None = None,
    ) -> ReviewDecision:
        targets = tuple(sorted(set(event_ids)))
        if len(targets) < 2:
            raise ReviewDecisionError("grouping events requires at least two event ids")
        before = self._event_group_snapshot(targets)
        return create_decision(
            ReviewAction.GROUP_EVENTS,
            targets,
            payload={"before": before},
            reason=reason,
            supersedes=supersedes,
        )

    def ungroup_event(
        self,
        event_id: str,
        *,
        supersedes: str,
        reason: str | None = None,
    ) -> ReviewDecision:
        return create_decision(
            ReviewAction.UNGROUP_EVENT,
            (event_id,),
            reason=reason,
            supersedes=supersedes,
        )

    def apply(self, decisions: Iterable[ReviewDecision] | ReviewDecision) -> SnapshotT:
        items = (decisions,) if isinstance(decisions, ReviewDecision) else tuple(decisions)
        return self._case_factory(self._engine.apply(self._data, items))

    def _entity_merge_snapshot(self, target_ids: tuple[str, ...]) -> dict[str, Any]:
        entities = tuple(entity for entity in self._data.entities if entity.id in target_ids)
        if {entity.id for entity in entities} != set(target_ids):
            missing = sorted(set(target_ids) - {entity.id for entity in entities})
            raise ReviewDecisionError(f"unknown entity ids: {', '.join(missing)}")
        claims = tuple(
            claim
            for claim in self._data.claims
            if any(participant.entity_id in target_ids for participant in claim.participants)
        )
        events = tuple(
            event
            for event in self._data.events
            if any(entity_id in target_ids for entity_id in event.participant_entity_ids)
        )
        relationships = tuple(
            relationship
            for relationship in self._data.relationships
            if relationship.source_id in target_ids or relationship.target_id in target_ids
        )
        return _snapshot(
            entities=entities,
            claims=claims,
            events=events,
            relationships=relationships,
        )

    def _event_group_snapshot(self, target_ids: tuple[str, ...]) -> dict[str, Any]:
        events = tuple(event for event in self._data.events if event.id in target_ids)
        if {event.id for event in events} != set(target_ids):
            missing = sorted(set(target_ids) - {event.id for event in events})
            raise ReviewDecisionError(f"unknown event ids: {', '.join(missing)}")
        relationships = tuple(
            relationship
            for relationship in self._data.relationships
            if relationship.source_id in target_ids or relationship.target_id in target_ids
        )
        return _snapshot(events=events, relationships=relationships)


def _snapshot(**collections: tuple[Any, ...]) -> dict[str, Any]:
    return {
        name: [item.model_dump(mode="json") for item in sorted(items, key=lambda item: item.id)]
        for name, items in sorted(collections.items())
    }


def _identity(data: CaseData) -> CaseData:
    return data
