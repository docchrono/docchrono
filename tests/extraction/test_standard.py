from __future__ import annotations

from docchrono.domain import (
    CaseConfig,
    ClaimKind,
    Document,
    DocumentSegment,
    Modality,
    OffsetMap,
    Polarity,
    TemporalPrecision,
)
from docchrono.extraction import ExtractionEngine, TemporalNormalizer


def document(
    text: str,
    *,
    metadata: dict[str, object] | None = None,
    segments: tuple[DocumentSegment, ...] = (),
) -> Document:
    return Document(
        id="document_standard",
        content_sha256="1" * 64,
        source_reference_ids=("source_standard",),
        media_type="text/plain",
        raw_text=text,
        normalized_text=text,
        normalized_to_raw=OffsetMap.identity(len(text)),
        segments=segments,
        metadata=metadata or {},
        parser_name="test",
        parser_version="1",
    )


def test_negation_modality_and_pronoun_abstention() -> None:
    source = document(
        "Alice Smith did not approve Contract #A-123. "
        "Robert Williams may approve the payment. He approved it."
    )

    result = ExtractionEngine.standard().extract([source])
    event_claims = [claim for claim in result.claims if claim.kind == ClaimKind.EVENT]

    assert len(event_claims) == 2
    by_actor = {
        next(
            participant.mention_id for participant in claim.participants if participant.mention_id
        ): claim
        for claim in event_claims
    }
    assert {claim.polarity for claim in by_actor.values()} == {
        Polarity.NEGATED,
        Polarity.AFFIRMED,
    }
    assert {claim.modality for claim in by_actor.values()} == {
        Modality.ASSERTED,
        Modality.POSSIBLE,
    }
    assert all(
        mention.text.casefold() != "he" for mention in result.mentions if mention.id in by_actor
    )


def test_cooccurrence_does_not_create_a_semantic_relationship() -> None:
    result = ExtractionEngine.standard().extract(
        [document("Alice Smith and Acme Corporation attended the conference.")]
    )

    assert not [claim for claim in result.claims if claim.kind == ClaimKind.RELATIONSHIP]


def test_relative_date_requires_explicit_document_reference() -> None:
    unresolved = TemporalNormalizer().find(document("The filing happened yesterday."))[0]
    resolved = TemporalNormalizer().find(
        document(
            "The filing happened yesterday.",
            metadata={"reference_datetime": "2026-08-14T12:00:00"},
        )
    )[0]

    assert unresolved.expression.precision == TemporalPrecision.UNRESOLVED
    assert not unresolved.expression.resolved
    assert unresolved.expression.original_text == "yesterday"
    assert resolved.expression.resolved
    assert resolved.expression.start == "2026-08-13"
    assert resolved.expression.is_relative


def test_incomplete_metadata_date_never_uses_the_wall_clock() -> None:
    candidate = TemporalNormalizer().find(
        document(
            "The filing happened yesterday.",
            metadata={"reference_datetime": "March 3"},
        )
    )[0]

    assert not candidate.expression.resolved
    assert candidate.expression.precision == TemporalPrecision.UNRESOLVED


def test_month_precision_is_preserved_as_an_interval() -> None:
    candidate = TemporalNormalizer().find(document("The term begins in August 2026."))[0]

    assert candidate.expression.precision == TemporalPrecision.MONTH
    assert candidate.expression.start == "2026-08-01"
    assert candidate.expression.end == "2026-08-31"


def test_email_headers_create_structural_claims() -> None:
    from_value = "Alice Smith <alice@example.com>"
    to_value = "Bob Jones <bob@example.com>"
    date_value = "Friday, 14 August 2026 10:00:00 -0500"
    text = f"{from_value}\n{to_value}\n{date_value}\nQuarterly update\nHello Bob"
    from_start = 0
    to_start = len(from_value) + 1
    date_start = to_start + len(to_value) + 1
    segments = (
        DocumentSegment(kind="field", field="from", raw_start=from_start, raw_end=len(from_value)),
        DocumentSegment(
            kind="field",
            field="to",
            raw_start=to_start,
            raw_end=to_start + len(to_value),
        ),
        DocumentSegment(
            kind="field",
            field="date",
            raw_start=date_start,
            raw_end=date_start + len(date_value),
        ),
    )
    source = document(
        text,
        metadata={
            "headers": {
                "from": (from_value,),
                "to": (to_value,),
                "date": (date_value,),
                "subject": ("Quarterly update",),
            }
        },
        segments=segments,
    )

    result = ExtractionEngine.standard().extract([source], CaseConfig(default_timezone="UTC"))
    predicates = {claim.predicate for claim in result.claims}

    assert {"SENT", "RECEIVED", "EMAIL_SENT"} <= predicates
    assert all(
        claim.polarity == Polarity.AFFIRMED and claim.modality == Modality.ASSERTED
        for claim in result.claims
        if claim.predicate in {"SENT", "RECEIVED", "EMAIL_SENT"}
    )
    assert any(
        expression.original_text == date_value and expression.resolved
        for expression in result.temporal_expressions
    )
