from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol, TypeVar, cast

from docchrono.domain import (
    CaseData,
    Claim,
    Entity,
    Event,
    Mention,
    Relationship,
    ReviewAction,
    ReviewDecision,
    TemporalExpression,
)
from docchrono.domain.ids import stable_id
from docchrono.errors import ReviewDecisionError
from docchrono.extraction import ExtractionResult
from docchrono.resolution import EntityResolver
from docchrono.review.decisions import expected_decision_id


class ReviewEngine:
    """Validate and replay review logs into new immutable ``CaseData`` snapshots."""

    def apply(
        self,
        data: CaseData,
        decisions: Iterable[ReviewDecision],
    ) -> CaseData:
        log = _append_unique(data.review_decisions, tuple(decisions))
        self.effective_decisions(log)
        existing_ids = {decision.id for decision in data.review_decisions}
        result = data
        decision_map = {decision.id: decision for decision in log}
        for decision in log:
            if decision.id not in existing_ids:
                result = self._apply_one(result, decision, decision_map)
        return result.model_copy(update={"review_decisions": log})

    def replay(
        self,
        base: CaseData,
        decisions: Iterable[ReviewDecision] | None = None,
    ) -> CaseData:
        if base.review_decisions and decisions is None:
            raise ReviewDecisionError(
                "replay requires unreviewed base data when no explicit decision log is supplied"
            )
        log = tuple(decisions) if decisions is not None else base.review_decisions
        log = _append_unique((), log)
        self.effective_decisions(log)
        result = base.model_copy(update={"review_decisions": ()})
        decision_map = {decision.id: decision for decision in log}
        for decision in log:
            result = self._apply_one(result, decision, decision_map)
        return result.model_copy(update={"review_decisions": log})

    def effective_decisions(
        self,
        decisions: Iterable[ReviewDecision],
    ) -> tuple[ReviewDecision, ...]:
        active: dict[str, ReviewDecision] = {}
        seen: set[str] = set()
        for decision in decisions:
            self._validate_identity_and_shape(decision)
            if decision.id in seen:
                continue
            seen.add(decision.id)
            conflicts = tuple(
                current for current in active.values() if _decisions_conflict(current, decision)
            )
            if decision.supersedes is None:
                if conflicts:
                    ids = ", ".join(item.id for item in conflicts)
                    raise ReviewDecisionError(
                        f"decision {decision.id} conflicts with {ids}; set supersedes explicitly"
                    )
            else:
                superseded = active.get(decision.supersedes)
                if superseded is None:
                    raise ReviewDecisionError(
                        f"decision {decision.id} supersedes an unknown or inactive decision "
                        f"{decision.supersedes}"
                    )
                if not _decisions_conflict(superseded, decision):
                    raise ReviewDecisionError(
                        f"decision {decision.id} cannot supersede unrelated decision "
                        f"{superseded.id}"
                    )
                other_conflicts = tuple(item for item in conflicts if item.id != superseded.id)
                if other_conflicts:
                    ids = ", ".join(item.id for item in other_conflicts)
                    raise ReviewDecisionError(
                        f"decision {decision.id} also conflicts with active decisions {ids}"
                    )
                del active[superseded.id]
            active[decision.id] = decision
        return tuple(active.values())

    def _validate_identity_and_shape(self, decision: ReviewDecision) -> None:
        expected = expected_decision_id(decision)
        if decision.id != expected:
            raise ReviewDecisionError(
                f"review decision id {decision.id!r} is not its content-derived id {expected!r}"
            )
        if not decision.target_ids or len(set(decision.target_ids)) != len(decision.target_ids):
            raise ReviewDecisionError("review decision targets must be non-empty and unique")
        target_count = len(decision.target_ids)
        if decision.action in {ReviewAction.MERGE_ENTITIES, ReviewAction.GROUP_EVENTS}:
            if target_count < 2:
                raise ReviewDecisionError(f"{decision.action.value} requires at least two targets")
            before = decision.payload.get("before")
            if not isinstance(before, Mapping):
                raise ReviewDecisionError(f"{decision.action.value} requires an auditable snapshot")
        elif target_count != 1:
            raise ReviewDecisionError(f"{decision.action.value} requires exactly one target")
        if (
            decision.action in {ReviewAction.UNMERGE_ENTITY, ReviewAction.UNGROUP_EVENT}
            and decision.supersedes is None
        ):
            raise ReviewDecisionError(
                f"{decision.action.value} must supersede its grouping decision"
            )
        if decision.action == ReviewAction.SET_CANONICAL_NAME:
            name = decision.payload.get("canonical_name")
            if not isinstance(name, str) or not name.strip():
                raise ReviewDecisionError("SET_CANONICAL_NAME requires a non-blank canonical_name")

    def _apply_one(
        self,
        data: CaseData,
        decision: ReviewDecision,
        decisions: Mapping[str, ReviewDecision],
    ) -> CaseData:
        match decision.action:
            case ReviewAction.ACCEPT:
                return _apply_acceptance(data, decision, accepted=True)
            case ReviewAction.REJECT:
                return _apply_acceptance(data, decision, accepted=False)
            case ReviewAction.MERGE_ENTITIES:
                return _merge_entities(data, decision)
            case ReviewAction.UNMERGE_ENTITY:
                return _unmerge_entity(data, decision, decisions)
            case ReviewAction.SET_CANONICAL_NAME:
                return _set_canonical_name(data, decision)
            case ReviewAction.GROUP_EVENTS:
                return _group_events(data, decision)
            case ReviewAction.UNGROUP_EVENT:
                return _ungroup_event(data, decision, decisions)
        raise ReviewDecisionError(f"unsupported review action: {decision.action}")


def merged_entity_id(target_ids: Iterable[str]) -> str:
    return stable_id("entity", "review-merge", tuple(sorted(set(target_ids))))


def grouped_event_id(target_ids: Iterable[str]) -> str:
    return stable_id("event", "review-group", tuple(sorted(set(target_ids))))


def _append_unique(
    existing: tuple[ReviewDecision, ...],
    additions: tuple[ReviewDecision, ...],
) -> tuple[ReviewDecision, ...]:
    result = list(existing)
    by_id = {decision.id: decision for decision in existing}
    for decision in additions:
        previous = by_id.get(decision.id)
        if previous is not None:
            if previous != decision:
                raise ReviewDecisionError(
                    f"duplicate decision id has different content: {decision.id}"
                )
            continue
        result.append(decision)
        by_id[decision.id] = decision
    return tuple(result)


def _decisions_conflict(left: ReviewDecision, right: ReviewDecision) -> bool:
    left_targets = set(left.target_ids)
    right_targets = set(right.target_ids)
    if left.id == right.id:
        return False
    if left.action in {ReviewAction.ACCEPT, ReviewAction.REJECT} and right.action in {
        ReviewAction.ACCEPT,
        ReviewAction.REJECT,
    }:
        return bool(left_targets & right_targets)
    if left.action == right.action == ReviewAction.SET_CANONICAL_NAME:
        return bool(left_targets & right_targets)
    if left.action == right.action == ReviewAction.MERGE_ENTITIES:
        return bool(left_targets & right_targets)
    if left.action == right.action == ReviewAction.GROUP_EVENTS:
        return bool(left_targets & right_targets)
    if left.action == right.action == ReviewAction.UNMERGE_ENTITY:
        return bool(left_targets & right_targets)
    if left.action == right.action == ReviewAction.UNGROUP_EVENT:
        return bool(left_targets & right_targets)
    inverse_pairs = {
        (ReviewAction.MERGE_ENTITIES, ReviewAction.UNMERGE_ENTITY),
        (ReviewAction.UNMERGE_ENTITY, ReviewAction.MERGE_ENTITIES),
        (ReviewAction.GROUP_EVENTS, ReviewAction.UNGROUP_EVENT),
        (ReviewAction.UNGROUP_EVENT, ReviewAction.GROUP_EVENTS),
    }
    return (left.action, right.action) in inverse_pairs and (
        left.id == right.supersedes or right.id == left.supersedes
    )


def _apply_acceptance(
    data: CaseData,
    decision: ReviewDecision,
    *,
    accepted: bool,
) -> CaseData:
    target_id = decision.target_ids[0]
    item = next((item for item in data.review_items if item.id == target_id), None)
    target_ids = set(item.target_ids) if item is not None else {target_id}
    mention_ids = target_ids & {mention.id for mention in data.mentions}
    claim_ids = target_ids & {claim.id for claim in data.claims}
    if item is not None and item.kind == "entity_merge_candidate":
        raise ReviewDecisionError(
            "entity merge candidates must be resolved with merge_entities(), not accept/reject"
        )
    changed = False
    mentions: list[Mention] = []
    for mention in data.mentions:
        if mention.id in mention_ids:
            mentions.append(mention.model_copy(update={"accepted": accepted}))
            changed = True
        else:
            mentions.append(mention)
    claims: list[Claim] = []
    for claim in data.claims:
        if claim.id in claim_ids:
            claims.append(claim.model_copy(update={"accepted": accepted}))
            changed = True
        else:
            claims.append(claim)
    known_ids = _all_record_ids(data)
    if item is None and target_id not in known_ids:
        raise ReviewDecisionError(f"unknown accept/reject target: {target_id}")
    if not changed:
        raise ReviewDecisionError(f"accept/reject target has no reviewable record: {target_id}")

    updated_mentions = tuple(mentions)
    updated_claims = tuple(claims)
    if mention_ids:
        resolution = EntityResolver().resolve(
            ExtractionResult(
                evidence_spans=data.evidence_spans,
                mentions=updated_mentions,
                claims=updated_claims,
            ),
            config=data.config,
        )
        review_items = _unique_records((*data.review_items, *resolution.review_items))
        return data.model_copy(
            update={
                "mentions": updated_mentions,
                "claims": resolution.claims,
                "entities": resolution.entities,
                "events": resolution.events,
                "relationships": resolution.relationships,
                "review_items": review_items,
            }
        )

    events, relationships = EntityResolver.derive_views(updated_claims)
    return data.model_copy(
        update={"claims": updated_claims, "events": events, "relationships": relationships}
    )


def _merge_entities(data: CaseData, decision: ReviewDecision) -> CaseData:
    target_ids = set(decision.target_ids)
    entities = tuple(entity for entity in data.entities if entity.id in target_ids)
    if {entity.id for entity in entities} != target_ids:
        missing = sorted(target_ids - {entity.id for entity in entities})
        raise ReviewDecisionError(f"cannot merge unknown entity ids: {', '.join(missing)}")
    _verify_snapshot_entities(decision, entities)
    types = {entity.type for entity in entities}
    if len(types) != 1:
        raise ReviewDecisionError("entities with different types cannot be merged")

    canonical_payload = decision.payload.get("canonical_name")
    if canonical_payload is not None and (
        not isinstance(canonical_payload, str) or not canonical_payload.strip()
    ):
        raise ReviewDecisionError("canonical_name must be a non-blank string or null")
    canonical_name = (
        canonical_payload
        if isinstance(canonical_payload, str)
        else min(
            entities,
            key=lambda entity: (
                -len(entity.canonical_name),
                entity.canonical_name.casefold(),
                entity.canonical_name,
                entity.id,
            ),
        ).canonical_name
    )
    aliases = {
        alias
        for entity in entities
        for alias in (entity.canonical_name, *entity.aliases)
        if alias != canonical_name
    }
    merged_id = merged_entity_id(target_ids)
    if any(entity.id == merged_id and entity.id not in target_ids for entity in data.entities):
        raise ReviewDecisionError(f"merged entity id already exists: {merged_id}")
    merged = Entity(
        id=merged_id,
        type=entities[0].type,
        canonical_name=canonical_name,
        aliases=tuple(sorted(aliases, key=lambda value: (value.casefold(), value))),
        mention_ids=tuple(sorted({item for entity in entities for item in entity.mention_ids})),
        score=max(entity.score for entity in entities),
    )
    updated_entities = (
        *sorted(
            (entity for entity in data.entities if entity.id not in target_ids),
            key=lambda entity: entity.id,
        ),
        merged,
    )
    updated_entities = tuple(sorted(updated_entities, key=lambda entity: entity.id))

    claims = tuple(_remap_claim_entities(claim, target_ids, merged_id) for claim in data.claims)
    events = tuple(_remap_event_entities(event, target_ids, merged_id) for event in data.events)
    relationships = _coalesce_relationships(
        _remap_relationship_nodes(relationship, target_ids, merged_id)
        for relationship in data.relationships
    )
    return data.model_copy(
        update={
            "entities": updated_entities,
            "claims": claims,
            "events": events,
            "relationships": relationships,
        }
    )


def _unmerge_entity(
    data: CaseData,
    decision: ReviewDecision,
    decisions: Mapping[str, ReviewDecision],
) -> CaseData:
    source = decisions.get(decision.supersedes or "")
    if source is None or source.action != ReviewAction.MERGE_ENTITIES:
        raise ReviewDecisionError("UNMERGE_ENTITY must supersede a MERGE_ENTITIES decision")
    merged_id = merged_entity_id(source.target_ids)
    if decision.target_ids != (merged_id,):
        raise ReviewDecisionError(f"UNMERGE_ENTITY target must be merged entity id {merged_id}")
    if not any(entity.id == merged_id for entity in data.entities):
        raise ReviewDecisionError(f"merged entity does not exist: {merged_id}")

    entities = _snapshot_entities(source)
    claims = _snapshot_claims(source)
    events = _snapshot_events(source)
    relationships = _snapshot_relationships(source)
    remaining_relationships = tuple(
        relationship
        for relationship in data.relationships
        if relationship.source_id != merged_id and relationship.target_id != merged_id
    )
    return data.model_copy(
        update={
            "entities": _restore_by_id(data.entities, entities, remove_ids={merged_id}),
            "claims": _restore_by_id(data.claims, claims),
            "events": _restore_by_id(data.events, events),
            "relationships": _restore_by_id(remaining_relationships, relationships),
        }
    )


def _set_canonical_name(data: CaseData, decision: ReviewDecision) -> CaseData:
    target_id = decision.target_ids[0]
    canonical_name = decision.payload["canonical_name"]
    assert isinstance(canonical_name, str)
    found = False
    entities: list[Entity] = []
    for entity in data.entities:
        if entity.id != target_id:
            entities.append(entity)
            continue
        found = True
        aliases = set(entity.aliases)
        if entity.canonical_name != canonical_name:
            aliases.add(entity.canonical_name)
            aliases.discard(canonical_name)
        entities.append(
            entity.model_copy(
                update={
                    "canonical_name": canonical_name,
                    "aliases": tuple(sorted(aliases, key=lambda value: (value.casefold(), value))),
                }
            )
        )
    if not found:
        raise ReviewDecisionError(f"unknown entity id for canonical name: {target_id}")
    return data.model_copy(update={"entities": tuple(entities)})


def _group_events(data: CaseData, decision: ReviewDecision) -> CaseData:
    target_ids = set(decision.target_ids)
    events = tuple(event for event in data.events if event.id in target_ids)
    if {event.id for event in events} != target_ids:
        missing = sorted(target_ids - {event.id for event in events})
        raise ReviewDecisionError(f"cannot group unknown event ids: {', '.join(missing)}")
    _verify_snapshot_events(decision, events)
    event_types = {event.type for event in events}
    if len(event_types) != 1:
        raise ReviewDecisionError("events with different types cannot be grouped")

    group_id = grouped_event_id(target_ids)
    grouped = Event(
        id=group_id,
        type=events[0].type,
        title=" / ".join(sorted({event.title for event in events})),
        claim_ids=tuple(sorted({claim_id for event in events for claim_id in event.claim_ids})),
        participant_entity_ids=tuple(
            sorted({entity_id for event in events for entity_id in event.participant_entity_ids})
        ),
        temporal=_unique_temporals(events),
        score=max(event.score for event in events),
        provisional=False,
    )
    remaining = (event for event in data.events if event.id not in target_ids)
    updated_events = tuple(sorted((*remaining, grouped), key=lambda event: event.id))
    relationships = _coalesce_relationships(
        _remap_relationship_nodes(relationship, target_ids, group_id)
        for relationship in data.relationships
    )
    return data.model_copy(update={"events": updated_events, "relationships": relationships})


def _ungroup_event(
    data: CaseData,
    decision: ReviewDecision,
    decisions: Mapping[str, ReviewDecision],
) -> CaseData:
    source = decisions.get(decision.supersedes or "")
    if source is None or source.action != ReviewAction.GROUP_EVENTS:
        raise ReviewDecisionError("UNGROUP_EVENT must supersede a GROUP_EVENTS decision")
    group_id = grouped_event_id(source.target_ids)
    if decision.target_ids != (group_id,):
        raise ReviewDecisionError(f"UNGROUP_EVENT target must be grouped event id {group_id}")
    if not any(event.id == group_id for event in data.events):
        raise ReviewDecisionError(f"grouped event does not exist: {group_id}")
    events = _snapshot_events(source)
    relationships = _snapshot_relationships(source)
    remaining_relationships = tuple(
        relationship
        for relationship in data.relationships
        if relationship.source_id != group_id and relationship.target_id != group_id
    )
    return data.model_copy(
        update={
            "events": _restore_by_id(data.events, events, remove_ids={group_id}),
            "relationships": _restore_by_id(remaining_relationships, relationships),
        }
    )


def _remap_claim_entities(claim: Claim, old_ids: set[str], new_id: str) -> Claim:
    participants = tuple(
        participant.model_copy(update={"entity_id": new_id})
        if participant.entity_id in old_ids
        else participant
        for participant in claim.participants
    )
    return claim.model_copy(update={"participants": participants})


def _remap_event_entities(event: Event, old_ids: set[str], new_id: str) -> Event:
    participants = {
        new_id if entity_id in old_ids else entity_id for entity_id in event.participant_entity_ids
    }
    return event.model_copy(update={"participant_entity_ids": tuple(sorted(participants))})


def _remap_relationship_nodes(
    relationship: Relationship,
    old_ids: set[str],
    new_id: str,
) -> Relationship:
    source_id = new_id if relationship.source_id in old_ids else relationship.source_id
    target_id = new_id if relationship.target_id in old_ids else relationship.target_id
    return relationship.model_copy(
        update={
            "id": stable_id("relationship", source_id, target_id, relationship.type),
            "source_id": source_id,
            "target_id": target_id,
        }
    )


def _coalesce_relationships(
    relationships: Iterable[Relationship],
) -> tuple[Relationship, ...]:
    grouped: dict[tuple[str, str, str], list[Relationship]] = {}
    for relationship in relationships:
        key = (relationship.source_id, relationship.target_id, relationship.type)
        grouped.setdefault(key, []).append(relationship)
    result: list[Relationship] = []
    for (source_id, target_id, relationship_type), values in sorted(grouped.items()):
        result.append(
            Relationship(
                id=stable_id("relationship", source_id, target_id, relationship_type),
                source_id=source_id,
                target_id=target_id,
                type=relationship_type,
                supporting_claim_ids=tuple(
                    sorted({claim for value in values for claim in value.supporting_claim_ids})
                ),
                opposing_claim_ids=tuple(
                    sorted({claim for value in values for claim in value.opposing_claim_ids})
                ),
                score=max(value.score for value in values),
            )
        )
    return tuple(result)


class _HasId(Protocol):
    id: str


_RecordT = TypeVar("_RecordT", bound=_HasId)


def _unique_records(items: Iterable[_RecordT]) -> tuple[_RecordT, ...]:
    by_id = {item.id: item for item in items}
    return tuple(by_id[item_id] for item_id in sorted(by_id))


def _unique_temporals(events: tuple[Event, ...]) -> tuple[TemporalExpression, ...]:
    by_content: dict[str, TemporalExpression] = {}
    for event in events:
        for temporal in event.temporal:
            key = stable_id("temporal", temporal)
            by_content[key] = temporal
    return tuple(by_content[key] for key in sorted(by_content))


def _verify_snapshot_entities(decision: ReviewDecision, current: tuple[Entity, ...]) -> None:
    snapshot = _snapshot_entities(decision)
    if tuple(item.model_dump(mode="json") for item in snapshot) != tuple(
        item.model_dump(mode="json") for item in sorted(current, key=lambda item: item.id)
    ):
        raise ReviewDecisionError("entity merge snapshot does not match current entities")


def _verify_snapshot_events(decision: ReviewDecision, current: tuple[Event, ...]) -> None:
    snapshot = _snapshot_events(decision)
    if tuple(item.model_dump(mode="json") for item in snapshot) != tuple(
        item.model_dump(mode="json") for item in sorted(current, key=lambda item: item.id)
    ):
        raise ReviewDecisionError("event group snapshot does not match current events")


def _snapshot_entities(decision: ReviewDecision) -> tuple[Entity, ...]:
    return tuple(Entity.model_validate(item) for item in _snapshot_section(decision, "entities"))


def _snapshot_claims(decision: ReviewDecision) -> tuple[Claim, ...]:
    return tuple(Claim.model_validate(item) for item in _snapshot_section(decision, "claims"))


def _snapshot_events(decision: ReviewDecision) -> tuple[Event, ...]:
    return tuple(Event.model_validate(item) for item in _snapshot_section(decision, "events"))


def _snapshot_relationships(decision: ReviewDecision) -> tuple[Relationship, ...]:
    return tuple(
        Relationship.model_validate(item) for item in _snapshot_section(decision, "relationships")
    )


def _snapshot_section(decision: ReviewDecision, name: str) -> tuple[object, ...]:
    before = decision.payload.get("before")
    if not isinstance(before, Mapping):
        raise ReviewDecisionError(f"decision {decision.id} has no before snapshot")
    typed_before = cast("Mapping[str, object]", before)
    raw = typed_before.get(name, ())
    if not isinstance(raw, (list, tuple)):
        raise ReviewDecisionError(f"decision {decision.id} snapshot section {name!r} is invalid")
    return tuple(cast("list[object] | tuple[object, ...]", raw))


def _restore_by_id(
    current: tuple[Any, ...],
    originals: tuple[Any, ...],
    *,
    remove_ids: set[str] | None = None,
) -> tuple[Any, ...]:
    original_ids = {item.id for item in originals}
    removed = remove_ids or set()
    combined = [item for item in current if item.id not in original_ids and item.id not in removed]
    combined.extend(originals)
    return tuple(sorted(combined, key=lambda item: item.id))


def _all_record_ids(data: CaseData) -> set[str]:
    collections = (
        data.source_references,
        data.documents,
        data.evidence_spans,
        data.mentions,
        data.claims,
        data.entities,
        data.events,
        data.relationships,
        data.review_items,
    )
    return {item.id for collection in collections for item in collection}
