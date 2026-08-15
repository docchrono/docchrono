from __future__ import annotations

from dataclasses import dataclass

import pytest

from docchrono.domain import CaseConfig, Document, EntityType, OffsetMap, TemporalExpression
from docchrono.errors import AdapterContractError
from docchrono.extraction import (
    ExtractionBatch,
    ExtractionContext,
    ExtractionEngine,
    ExtractorRegistry,
    MentionCandidate,
    TemporalCandidate,
)


def document(
    text: str,
    *,
    raw_text: str | None = None,
    offset_map: tuple[int, ...] | None = None,
    metadata: dict[str, object] | None = None,
) -> Document:
    raw = raw_text if raw_text is not None else text
    mapping = offset_map if offset_map is not None else tuple(range(len(text) + 1))
    return Document(
        id="document_test",
        content_sha256="0" * 64,
        source_reference_ids=("source_test",),
        media_type="text/plain",
        raw_text=raw,
        normalized_text=text,
        normalized_to_raw=OffsetMap.from_dense(mapping),
        metadata=metadata or {},
        parser_name="test",
        parser_version="1",
    )


@dataclass(frozen=True)
class StaticExtractor:
    name: str
    version: str
    order: int
    batch: ExtractionBatch

    def extract(self, document: Document, context: ExtractionContext) -> ExtractionBatch:
        del document, context
        return self.batch


def test_registry_order_is_deterministic() -> None:
    later = StaticExtractor("zeta", "1", 20, ExtractionBatch())
    lexical_second = StaticExtractor("beta", "1", 10, ExtractionBatch())
    lexical_first = StaticExtractor("alpha", "2", 10, ExtractionBatch())

    registry = ExtractorRegistry((later, lexical_second, lexical_first))

    assert [item.name for item in registry.extractors] == ["alpha", "beta", "zeta"]


def test_adapter_cannot_emit_an_unmapped_span() -> None:
    adapter = StaticExtractor(
        "invalid",
        "1",
        1,
        ExtractionBatch(
            mentions=(
                MentionCandidate(
                    key="bad",
                    normalized_start=0,
                    normalized_end=100,
                    entity_type=EntityType.PERSON,
                    score=1.0,
                ),
            )
        ),
    )
    engine = ExtractionEngine(ExtractorRegistry((adapter,)))

    with pytest.raises(AdapterContractError, match="outside document"):
        engine.extract([document("Alice")])


def test_evidence_uses_normalized_to_raw_boundaries_exactly() -> None:
    adapter = StaticExtractor(
        "mapped",
        "1",
        1,
        ExtractionBatch(
            mentions=(
                MentionCandidate(
                    key="smith",
                    normalized_start=6,
                    normalized_end=11,
                    entity_type=EntityType.PERSON,
                    score=0.95,
                ),
            )
        ),
    )
    engine = ExtractionEngine(ExtractorRegistry((adapter,)))
    source = document(
        "Alice Smith",
        raw_text="Alice  Smith",
        offset_map=(0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12),
    )

    result = engine.extract([source])

    span = result.evidence_spans[0]
    assert (span.raw_start, span.raw_end, span.quote) == (7, 12, "Smith")
    assert (span.normalized_start, span.normalized_end) == (6, 11)


def test_threshold_bands_retain_every_candidate() -> None:
    adapter = StaticExtractor(
        "scores",
        "1",
        1,
        ExtractionBatch(
            mentions=(
                MentionCandidate("accepted", 0, 1, EntityType.OTHER, 0.95),
                MentionCandidate("review", 1, 2, EntityType.OTHER, 0.75),
                MentionCandidate("deferred", 2, 3, EntityType.OTHER, 0.40),
            )
        ),
    )
    engine = ExtractionEngine(ExtractorRegistry((adapter,)))

    result = engine.extract(
        [document("ABC")],
        CaseConfig(auto_accept_threshold=0.90, review_threshold=0.60),
    )

    assert len(result.mentions) == 3
    assert len(result.accepted_mention_ids) == 1
    assert len(result.review_mention_ids) == 1
    assert len(result.deferred_mention_ids) == 1
    assert {item.kind for item in result.review_items} == {
        "mention_candidate",
        "mention_candidate_deferred",
    }


def test_temporal_original_text_must_match_candidate_span() -> None:
    adapter = StaticExtractor(
        "temporal",
        "1",
        1,
        ExtractionBatch(
            temporals=(
                TemporalCandidate(
                    0,
                    4,
                    TemporalExpression(original_text="not 2026"),
                    0.9,
                ),
            )
        ),
    )

    with pytest.raises(AdapterContractError, match="original_text"):
        ExtractionEngine(ExtractorRegistry((adapter,))).extract([document("2026")])
