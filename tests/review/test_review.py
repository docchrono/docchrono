import pytest

from docchrono.case import Case
from docchrono.domain import (
    BuildManifest,
    BuildReport,
    BuildStage,
    CaseConfig,
    CaseData,
    Claim,
    ClaimKind,
    ClaimParticipant,
    Entity,
    EntityType,
    Event,
    EventType,
    EvidenceSpan,
    Mention,
    Modality,
    Polarity,
    Relationship,
    ReviewItem,
    TemporalExpression,
    TemporalPrecision,
)
from docchrono.errors import ReviewDecisionError
from docchrono.review import ReviewBook, ReviewEngine, grouped_event_id, merged_entity_id


def _data() -> CaseData:
    first = Entity(
        id="entity-a",
        type=EntityType.PERSON,
        canonical_name="Alice Example",
        aliases=("Alice",),
        mention_ids=("mention-a",),
        score=0.8,
    )
    second = Entity(
        id="entity-b",
        type=EntityType.PERSON,
        canonical_name="A. Example",
        aliases=(),
        mention_ids=("mention-b",),
        score=0.9,
    )
    claim = Claim(
        id="claim-a",
        kind=ClaimKind.EVENT,
        predicate="APPROVED",
        participants=(ClaimParticipant(role="actor", entity_id=first.id),),
        evidence_span_ids=("evidence-a",),
        polarity=Polarity.AFFIRMED,
        modality=Modality.ASSERTED,
        extractor_name="fixture",
        extractor_version="1",
        score=0.9,
    )
    first_event = Event(
        id="event-a",
        type=EventType.ACTION,
        title="Approval",
        claim_ids=(claim.id,),
        participant_entity_ids=(second.id,),
        temporal=(
            TemporalExpression(
                original_text="January 1",
                start="2024-01-01",
                precision=TemporalPrecision.DAY,
                resolved=True,
            ),
        ),
        score=0.8,
    )
    second_event = Event(
        id="event-b",
        type=EventType.ACTION,
        title="The approval",
        claim_ids=(claim.id,),
        participant_entity_ids=(first.id,),
        temporal=(
            TemporalExpression(
                original_text="January 2",
                start="2024-01-02",
                precision=TemporalPrecision.DAY,
                resolved=True,
            ),
        ),
        score=0.9,
    )
    relationship = Relationship(
        id="relationship-a",
        source_id=first.id,
        target_id=first_event.id,
        type="PARTICIPATED_IN",
        supporting_claim_ids=(claim.id,),
        score=0.9,
    )
    review_item = ReviewItem(
        id="review-item-a",
        kind="claim",
        target_ids=(claim.id,),
        reason="Check assertion",
    )
    manifest = BuildManifest(
        docchrono_version="0.1.0",
        python_version="3.13",
        platform="test",
        config_fingerprint="fixture",
    )
    report = BuildReport(
        requested_stage=BuildStage.GRAPH,
        completed_stage=BuildStage.GRAPH,
        review_items=(review_item,),
        manifest=manifest,
    )
    return CaseData(
        config=CaseConfig(),
        claims=(claim,),
        entities=(first, second),
        events=(first_event, second_event),
        relationships=(relationship,),
        review_items=(review_item,),
        report=report,
    )


def test_merge_replays_deterministically_and_unmerge_restores_records() -> None:
    base = _data()
    merge = ReviewBook(base).merge_entities(
        ("entity-b", "entity-a"), canonical_name="Alice Example", reason="Same person"
    )

    merged = ReviewBook(base).apply(merge)
    result_id = merged_entity_id(("entity-a", "entity-b"))

    assert base.entities != merged.entities
    assert tuple(entity.id for entity in merged.entities) == (result_id,)
    assert merged.claims[0].participants[0].entity_id == result_id
    assert all(event.participant_entity_ids == (result_id,) for event in merged.events)
    assert merged.relationships[0].source_id == result_id
    assert ReviewEngine().replay(base, (merge,)) == merged

    unmerge = ReviewBook(merged).unmerge_entity(result_id, supersedes=merge.id)
    restored = ReviewBook(merged).apply(unmerge)

    assert restored.entities == base.entities
    assert restored.claims == base.claims
    assert restored.events == base.events
    assert restored.relationships == base.relationships
    assert len(restored.review_decisions) == 2


def test_group_preserves_competing_dates_and_ungroup_restores_events() -> None:
    base = _data()
    group = ReviewBook(base).group_events(("event-b", "event-a"), reason="Same occurrence")
    grouped = ReviewBook(base).apply(group)
    result_id = grouped_event_id(("event-a", "event-b"))

    assert tuple(event.id for event in grouped.events) == (result_id,)
    assert {temporal.start for temporal in grouped.events[0].temporal} == {
        "2024-01-01",
        "2024-01-02",
    }
    assert ReviewEngine().replay(base, (group,)) == grouped

    ungroup = ReviewBook(grouped).ungroup_event(result_id, supersedes=group.id)
    restored = ReviewBook(grouped).apply(ungroup)
    assert restored.events == base.events
    assert restored.relationships == base.relationships


def test_accept_reject_conflict_requires_explicit_supersession() -> None:
    base = _data()
    book = ReviewBook(base)
    accept = book.accept("review-item-a", reason="Supported")
    accepted = book.apply(accept)

    assert accepted.claims[0].accepted is True
    assert accepted.events
    assert accepted.relationships
    assert ReviewBook(accepted).pending == ()

    reject_without_link = ReviewBook(accepted).reject("review-item-a", reason="Changed mind")
    with pytest.raises(ReviewDecisionError, match="supersedes"):
        ReviewBook(accepted).apply(reject_without_link)

    reject = ReviewBook(accepted).reject(
        "review-item-a", reason="Contradicted", supersedes=accept.id
    )
    rejected = ReviewBook(accepted).apply(reject)
    assert rejected.claims[0].accepted is False
    assert rejected.events == ()
    assert rejected.relationships == ()
    assert ReviewEngine().replay(base, (accept, reject)) == rejected


def test_case_review_apply_returns_a_new_case_snapshot() -> None:
    original = Case(_data())
    decision = original.review.accept("review-item-a")

    reviewed = original.review.apply(decision)

    assert isinstance(reviewed, Case)
    assert reviewed is not original
    assert original.claims[0].accepted is False
    assert reviewed.claims[0].accepted is True


def test_accepting_and_rejecting_a_mention_rebuilds_entities() -> None:
    base = _data()
    evidence = EvidenceSpan(
        id="evidence-mention",
        document_id="document-a",
        raw_start=0,
        raw_end=11,
        normalized_start=0,
        normalized_end=11,
        quote="Carol Jones",
    )
    mention = Mention(
        id="mention-carol",
        evidence_span_id=evidence.id,
        entity_type=EntityType.PERSON,
        text="Carol Jones",
        normalized_text="carol jones",
        extractor_name="fixture",
        extractor_version="1",
        score=0.75,
        accepted=False,
    )
    item = ReviewItem(
        id="review-mention",
        kind="mention_candidate",
        target_ids=(mention.id,),
        reason="Check person mention",
        evidence_span_ids=(evidence.id,),
    )
    data = CaseData(
        config=base.config,
        evidence_spans=(evidence,),
        mentions=(mention,),
        review_items=(item,),
        report=base.report,
    )
    accept = ReviewBook(data).accept(item.id)

    accepted = ReviewBook(data).apply(accept)

    assert accepted.mentions[0].accepted
    assert tuple(entity.canonical_name for entity in accepted.entities) == ("Carol Jones",)

    reject = ReviewBook(accepted).reject(item.id, supersedes=accept.id)
    rejected = ReviewBook(accepted).apply(reject)
    assert not rejected.mentions[0].accepted
    assert rejected.entities == ()
