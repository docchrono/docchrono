from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from docchrono.domain import (
    CaseConfig,
    Claim,
    ClaimKind,
    Document,
    Entity,
    EntityType,
    Event,
    EvidenceSpan,
    Mention,
    Modality,
    Polarity,
    Relationship,
    ReviewItem,
    TemporalExpression,
)


@dataclass(frozen=True, slots=True)
class MentionCandidate:
    """A provisional mention expressed only in normalized-document offsets."""

    key: str
    normalized_start: int
    normalized_end: int
    entity_type: EntityType
    score: float
    normalized_text: str | None = None


@dataclass(frozen=True, slots=True)
class TemporalCandidate:
    """A temporal expression candidate before evidence provenance is attached."""

    normalized_start: int
    normalized_end: int
    expression: TemporalExpression
    score: float


@dataclass(frozen=True, slots=True)
class CandidateParticipant:
    role: str
    mention_key: str | None = None
    literal: str | None = None


@dataclass(frozen=True, slots=True)
class ClaimCandidate:
    """A source-scoped claim candidate produced by an extractor adapter."""

    key: str
    kind: ClaimKind
    predicate: str
    normalized_start: int
    normalized_end: int
    participants: tuple[CandidateParticipant, ...]
    score: float
    polarity: Polarity = Polarity.UNKNOWN
    modality: Modality = Modality.UNKNOWN
    temporal_keys: tuple[int, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtractionBatch:
    mentions: tuple[MentionCandidate, ...] = ()
    temporals: tuple[TemporalCandidate, ...] = ()
    claims: tuple[ClaimCandidate, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtractionContext:
    config: CaseConfig
    nlp: object


@runtime_checkable
class Extractor(Protocol):
    """Advanced extraction seam.

    Adapters return candidates in normalized offsets. The orchestration layer owns
    identifiers, raw offset mapping, threshold policy, and provenance validation.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def order(self) -> int: ...

    def extract(self, document: Document, context: ExtractionContext) -> ExtractionBatch: ...


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    evidence_spans: tuple[EvidenceSpan, ...] = ()
    mentions: tuple[Mention, ...] = ()
    temporal_expressions: tuple[TemporalExpression, ...] = ()
    claims: tuple[Claim, ...] = ()
    review_items: tuple[ReviewItem, ...] = ()
    accepted_mention_ids: tuple[str, ...] = ()
    review_mention_ids: tuple[str, ...] = ()
    deferred_mention_ids: tuple[str, ...] = ()
    accepted_claim_ids: tuple[str, ...] = ()
    review_claim_ids: tuple[str, ...] = ()
    deferred_claim_ids: tuple[str, ...] = ()
    adapter_versions: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class ResolutionCandidate:
    left_mention_id: str
    right_mention_id: str
    score: float
    reasons: tuple[str, ...]
    auto_merged: bool = False


@dataclass(frozen=True, slots=True)
class ResolutionResult:
    entities: tuple[Entity, ...] = ()
    claims: tuple[Claim, ...] = ()
    events: tuple[Event, ...] = ()
    relationships: tuple[Relationship, ...] = ()
    review_items: tuple[ReviewItem, ...] = ()
    candidates: tuple[ResolutionCandidate, ...] = ()
    mention_to_entity: tuple[tuple[str, str], ...] = ()
