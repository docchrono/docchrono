from __future__ import annotations

from docchrono.domain import (
    CaseConfig,
    Claim,
    ClaimKind,
    ClaimParticipant,
    EntityType,
    EvidenceSpan,
    Mention,
    Modality,
    Polarity,
)
from docchrono.extraction import ExtractionResult
from docchrono.resolution import EntityResolver


def span(index: int, document_id: str = "document_one") -> EvidenceSpan:
    return EvidenceSpan(
        id=f"evidence_{index}",
        document_id=document_id,
        raw_start=index,
        raw_end=index + 1,
        normalized_start=index,
        normalized_end=index + 1,
        quote="x",
    )


def mention(
    index: int,
    text: str,
    entity_type: EntityType,
    *,
    normalized: str | None = None,
    document_id: str = "document_one",
) -> tuple[EvidenceSpan, Mention]:
    evidence = span(index, document_id)
    return evidence, Mention(
        id=f"mention_{index}",
        evidence_span_id=evidence.id,
        entity_type=entity_type,
        text=text,
        normalized_text=normalized or text.casefold(),
        extractor_name="test",
        extractor_version="1",
        score=0.95,
    )


def test_exact_alias_merges_within_type_but_type_blocking_is_strict() -> None:
    evidence1, person1 = mention(1, "John Smith", EntityType.PERSON)
    evidence2, person2 = mention(2, "JOHN SMITH", EntityType.PERSON, document_id="document_two")
    evidence3, organization = mention(3, "John Smith", EntityType.ORGANIZATION)
    extraction = ExtractionResult(
        evidence_spans=(evidence1, evidence2, evidence3),
        mentions=(person1, person2, organization),
    )

    result = EntityResolver().resolve(extraction)

    assert len(result.entities) == 2
    person_entity = next(item for item in result.entities if item.type == EntityType.PERSON)
    assert person_entity.mention_ids == ("mention_1", "mention_2")
    assert next(
        item for item in result.entities if item.type == EntityType.ORGANIZATION
    ).mention_ids == ("mention_3",)


def test_exact_identifier_auto_merges_but_different_identifiers_do_not() -> None:
    values = [
        mention(1, "alice@example.com", EntityType.OTHER, normalized="email:alice@example.com"),
        mention(
            2,
            "ALICE@example.com",
            EntityType.OTHER,
            normalized="email:alice@example.com",
            document_id="document_two",
        ),
        mention(3, "bob@example.com", EntityType.OTHER, normalized="email:bob@example.com"),
    ]
    extraction = ExtractionResult(
        evidence_spans=tuple(item[0] for item in values),
        mentions=tuple(item[1] for item in values),
    )

    result = EntityResolver().resolve(extraction)

    assert len(result.entities) == 2
    assert any(candidate.score == 1.0 and candidate.auto_merged for candidate in result.candidates)
    assert all(candidate.score > 0.0 for candidate in result.candidates)


def test_long_organization_name_merges_with_its_acronym() -> None:
    evidence1, long_name = mention(1, "International Business Machines", EntityType.ORGANIZATION)
    evidence2, acronym = mention(
        2,
        "IBM",
        EntityType.ORGANIZATION,
        document_id="document_two",
    )
    extraction = ExtractionResult(
        evidence_spans=(evidence1, evidence2),
        mentions=(long_name, acronym),
    )

    result = EntityResolver().resolve(extraction)

    assert len(result.entities) == 1
    assert result.candidates[0].score == 0.99


def test_fuzzy_pair_is_explainable_and_never_auto_merges_by_default() -> None:
    evidence1, left = mention(1, "Robert Williams", EntityType.PERSON)
    evidence2, right = mention(2, "Robbert Williams", EntityType.PERSON, document_id="document_two")
    extraction = ExtractionResult(evidence_spans=(evidence1, evidence2), mentions=(left, right))

    result = EntityResolver().resolve(
        extraction,
        CaseConfig(review_threshold=0.60, auto_merge_threshold=0.98),
    )

    assert len(result.entities) == 2
    candidate = result.candidates[0]
    assert 0.60 <= candidate.score < 0.98
    assert not candidate.auto_merged
    assert "RapidFuzz" in candidate.reasons[0]
    assert result.review_items[0].kind == "entity_merge_candidate"


def test_relationships_derive_only_from_affirmed_asserted_claims() -> None:
    evidence1, alice = mention(1, "Alice Smith", EntityType.PERSON)
    evidence2, acme = mention(2, "Acme Corporation", EntityType.ORGANIZATION)
    participants = (
        ClaimParticipant(role="source", mention_id=alice.id),
        ClaimParticipant(role="target", mention_id=acme.id),
    )
    claims = (
        Claim(
            id="claim_support",
            kind=ClaimKind.RELATIONSHIP,
            predicate="WORKS_FOR",
            participants=participants,
            evidence_span_ids=(evidence1.id,),
            polarity=Polarity.AFFIRMED,
            modality=Modality.ASSERTED,
            extractor_name="test",
            extractor_version="1",
            score=0.95,
            accepted=True,
        ),
        Claim(
            id="claim_opposes",
            kind=ClaimKind.RELATIONSHIP,
            predicate="WORKS_FOR",
            participants=participants,
            evidence_span_ids=(evidence1.id,),
            polarity=Polarity.NEGATED,
            modality=Modality.ASSERTED,
            extractor_name="test",
            extractor_version="1",
            score=0.95,
            accepted=True,
        ),
        Claim(
            id="claim_possible",
            kind=ClaimKind.RELATIONSHIP,
            predicate="OFFICER_OF",
            participants=participants,
            evidence_span_ids=(evidence1.id,),
            polarity=Polarity.AFFIRMED,
            modality=Modality.POSSIBLE,
            extractor_name="test",
            extractor_version="1",
            score=0.95,
            accepted=True,
        ),
    )
    extraction = ExtractionResult(
        evidence_spans=(evidence1, evidence2),
        mentions=(alice, acme),
        claims=claims,
    )

    result = EntityResolver().resolve(extraction)

    assert len(result.relationships) == 1
    relationship = result.relationships[0]
    assert relationship.type == "WORKS_FOR"
    assert relationship.supporting_claim_ids == ("claim_support",)
    assert relationship.opposing_claim_ids == ("claim_opposes",)


def test_nonaffirmative_event_claims_do_not_create_occurrences() -> None:
    evidence, alice = mention(1, "Alice Smith", EntityType.PERSON)
    claims = tuple(
        Claim(
            id=f"claim_{polarity.value}_{modality.value}",
            kind=ClaimKind.EVENT,
            predicate="APPROVED",
            participants=(ClaimParticipant(role="actor", mention_id=alice.id),),
            evidence_span_ids=(evidence.id,),
            polarity=polarity,
            modality=modality,
            extractor_name="test",
            extractor_version="1",
            score=0.95,
            accepted=True,
        )
        for polarity, modality in (
            (Polarity.NEGATED, Modality.ASSERTED),
            (Polarity.AFFIRMED, Modality.POSSIBLE),
            (Polarity.AFFIRMED, Modality.ALLEGED),
        )
    )
    extraction = ExtractionResult(evidence_spans=(evidence,), mentions=(alice,), claims=claims)

    result = EntityResolver().resolve(extraction)

    assert result.events == ()


def test_event_participants_are_connected_to_the_event_in_the_graph_model() -> None:
    evidence, alice = mention(1, "Alice Smith", EntityType.PERSON)
    claim = Claim(
        id="claim_event",
        kind=ClaimKind.EVENT,
        predicate="APPROVED",
        participants=(ClaimParticipant(role="actor", mention_id=alice.id),),
        evidence_span_ids=(evidence.id,),
        polarity=Polarity.AFFIRMED,
        modality=Modality.ASSERTED,
        extractor_name="test",
        extractor_version="1",
        score=0.95,
        accepted=True,
    )
    extraction = ExtractionResult(evidence_spans=(evidence,), mentions=(alice,), claims=(claim,))

    result = EntityResolver().resolve(extraction)

    assert len(result.events) == 1
    assert len(result.relationships) == 1
    relationship = result.relationships[0]
    assert relationship.type == "PARTICIPATED_IN"
    assert relationship.target_id == result.events[0].id
    assert relationship.supporting_claim_ids == (claim.id,)
