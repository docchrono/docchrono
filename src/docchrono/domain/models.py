from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Literal, cast, overload

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_serializer,
    field_validator,
    model_validator,
)

from docchrono.domain.enums import (
    BuildStage,
    ClaimKind,
    DocumentFailureCode,
    EntityType,
    EventType,
    Modality,
    Polarity,
    ReviewAction,
    TemporalPrecision,
)
from docchrono.domain.frozen import freeze_mapping, thaw_mapping


class FrozenModel(BaseModel):
    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


@dataclass(frozen=True, slots=True)
class OffsetMap(Sequence[int]):
    """Compact immutable normalized-boundary to raw-boundary mapping.

    Each transition stores the point at which ``raw_boundary - normalized_boundary``
    changes. Indexing and iteration intentionally match a dense integer sequence.
    """

    normalized_length: int
    transitions: tuple[tuple[int, int], ...] = ((0, 0),)

    @classmethod
    def from_dense(cls, boundaries: Sequence[int]) -> OffsetMap:
        values = tuple(boundaries)
        if not values:
            raise ValueError("offset map requires at least one boundary")
        transitions: list[tuple[int, int]] = []
        previous_delta: int | None = None
        for index, raw_boundary in enumerate(values):
            delta = raw_boundary - index
            if delta != previous_delta:
                transitions.append((index, delta))
                previous_delta = delta
        return cls(normalized_length=len(values) - 1, transitions=tuple(transitions))

    @classmethod
    def identity(cls, normalized_length: int) -> OffsetMap:
        return cls(normalized_length=normalized_length)

    @classmethod
    def from_json(cls, value: Mapping[str, object]) -> OffsetMap:
        normalized_length = value.get("normalized_length")
        transitions_value = value.get("transitions")
        if (
            not isinstance(normalized_length, int)
            or isinstance(normalized_length, bool)
            or not isinstance(transitions_value, (tuple, list))
        ):
            raise ValueError("saved offset map is malformed")
        raw_transitions = cast("tuple[object, ...] | list[object]", transitions_value)
        transitions: list[tuple[int, int]] = []
        for raw_transition in raw_transitions:
            if not isinstance(raw_transition, (tuple, list)):
                raise ValueError("saved offset-map transition is malformed")
            pair = cast("tuple[object, ...] | list[object]", raw_transition)
            if len(pair) != 2:
                raise ValueError("saved offset-map transition is malformed")
            index, delta = pair
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not isinstance(delta, int)
                or isinstance(delta, bool)
            ):
                raise ValueError("saved offset-map transition must contain integers")
            transitions.append((index, delta))
        return cls(normalized_length=normalized_length, transitions=tuple(transitions))

    def __post_init__(self) -> None:
        if self.normalized_length < 0:
            raise ValueError("normalized_length must be non-negative")
        if not self.transitions or self.transitions[0][0] != 0:
            raise ValueError("offset-map transitions must begin at boundary zero")
        previous_index = -1
        previous_delta = 0
        boundary_count = self.normalized_length + 1
        for index, delta in self.transitions:
            if index <= previous_index or index >= boundary_count:
                raise ValueError("offset-map transition indexes must be increasing and in range")
            if index + delta < 0:
                raise ValueError("offset-map transition produces a negative raw boundary")
            if index > 0 and index - 1 + previous_delta > index + delta:
                raise ValueError("offset-map transitions must produce monotonic boundaries")
            previous_index = index
            previous_delta = delta

    def __len__(self) -> int:
        return self.normalized_length + 1

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[int, ...]: ...

    def __getitem__(self, index: int | slice) -> int | tuple[int, ...]:
        if isinstance(index, slice):
            return tuple(self[position] for position in range(*index.indices(len(self))))
        normalized_index = index + len(self) if index < 0 else index
        if normalized_index < 0 or normalized_index >= len(self):
            raise IndexError("offset-map index out of range")
        low = 0
        high = len(self.transitions)
        while low < high:
            middle = (low + high) // 2
            if self.transitions[middle][0] <= normalized_index:
                low = middle + 1
            else:
                high = middle
        return normalized_index + self.transitions[low - 1][1]

    def __iter__(self) -> Iterator[int]:
        transition_index = 0
        delta = self.transitions[0][1]
        for index in range(len(self)):
            if (
                transition_index + 1 < len(self.transitions)
                and self.transitions[transition_index + 1][0] == index
            ):
                transition_index += 1
                delta = self.transitions[transition_index][1]
            yield index + delta

    def __eq__(self, other: object) -> bool:
        if isinstance(other, OffsetMap):
            return (
                self.normalized_length == other.normalized_length
                and self.transitions == other.transitions
            )
        if isinstance(other, Sequence) and not isinstance(other, (str, bytes, bytearray)):
            sequence = cast("Sequence[object]", other)
            return len(self) == len(sequence) and all(
                boundary == sequence[index] for index, boundary in enumerate(self)
            )
        return False

    def __hash__(self) -> int:
        return hash((self.normalized_length, self.transitions))


class BoundingBox(FrozenModel):
    page: int = Field(ge=1)
    x0: float
    y0: float
    x1: float
    y1: float

    @model_validator(mode="after")
    def validate_bounds(self) -> BoundingBox:
        if self.x1 < self.x0 or self.y1 < self.y0:
            raise ValueError("bounding-box maximums must not precede minimums")
        return self


class DocumentSegment(FrozenModel):
    kind: Literal["page", "paragraph", "sentence", "field", "attachment", "sheet", "cell"]
    raw_start: int = Field(ge=0)
    raw_end: int = Field(ge=0)
    page: int | None = Field(default=None, ge=1)
    paragraph: int | None = Field(default=None, ge=0)
    sentence: int | None = Field(default=None, ge=0)
    field: str | None = None
    label: str | None = None

    @model_validator(mode="after")
    def validate_offsets(self) -> DocumentSegment:
        if self.raw_end < self.raw_start:
            raise ValueError("raw_end must be greater than or equal to raw_start")
        return self


class SourceReference(FrozenModel):
    id: str
    path: str
    filename: str
    media_type: str
    size_bytes: int = Field(ge=0)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    metadata: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("metadata", mode="after")
    @classmethod
    def freeze_metadata(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_mapping(value)

    @field_serializer("metadata")
    def serialize_metadata(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return thaw_mapping(value)


class Document(FrozenModel):
    id: str
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_reference_ids: tuple[str, ...]
    media_type: str
    raw_text: str
    normalized_text: str
    normalized_to_raw: OffsetMap
    segments: tuple[DocumentSegment, ...] = ()
    metadata: Mapping[str, Any] = Field(default_factory=dict)
    parser_name: str
    parser_version: str

    @field_validator("metadata", mode="after")
    @classmethod
    def freeze_metadata(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_mapping(value)

    @field_serializer("metadata")
    def serialize_metadata(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return thaw_mapping(value)

    @field_validator("normalized_to_raw", mode="before")
    @classmethod
    def compact_offset_map(cls, value: object) -> object:
        if isinstance(value, OffsetMap):
            return value
        if isinstance(value, Mapping):
            return OffsetMap.from_json(cast("Mapping[str, object]", value))
        if isinstance(value, (tuple, list)):
            items = cast("tuple[object, ...] | list[object]", value)
            if all(isinstance(item, int) and not isinstance(item, bool) for item in items):
                return OffsetMap.from_dense(cast("Sequence[int]", items))
        return cast("object", value)

    @field_serializer("normalized_to_raw")
    def serialize_offset_map(self, value: OffsetMap) -> dict[str, object]:
        return {
            "normalized_length": value.normalized_length,
            "transitions": [list(transition) for transition in value.transitions],
        }

    @model_validator(mode="after")
    def validate_offset_map(self) -> Document:
        if len(self.normalized_to_raw) != len(self.normalized_text) + 1:
            raise ValueError("normalized_to_raw must contain one boundary per normalized character")
        if not self.normalized_to_raw or self.normalized_to_raw[0] != 0:
            raise ValueError("normalized offset map must begin at raw boundary zero")
        if self.normalized_to_raw[-1] != len(self.raw_text):
            raise ValueError("normalized offset map must end at the raw-text boundary")
        if any(value < 0 for value in self.normalized_to_raw):
            raise ValueError("normalized offset map cannot contain negative boundaries")
        if any(left > right for left, right in pairwise(self.normalized_to_raw)):
            raise ValueError("normalized offset map must be monotonic")
        return self


class EvidenceSpan(FrozenModel):
    id: str
    document_id: str
    raw_start: int = Field(ge=0)
    raw_end: int = Field(ge=0)
    normalized_start: int | None = Field(default=None, ge=0)
    normalized_end: int | None = Field(default=None, ge=0)
    quote: str
    page: int | None = Field(default=None, ge=1)
    paragraph: int | None = Field(default=None, ge=0)
    sentence: int | None = Field(default=None, ge=0)
    field: str | None = None
    boxes: tuple[BoundingBox, ...] = ()

    @model_validator(mode="after")
    def validate_offsets(self) -> EvidenceSpan:
        if self.raw_end < self.raw_start:
            raise ValueError("raw_end must be greater than or equal to raw_start")
        if (self.normalized_start is None) != (self.normalized_end is None):
            raise ValueError("normalized offsets must be provided together")
        if (
            self.normalized_start is not None
            and self.normalized_end is not None
            and self.normalized_end < self.normalized_start
        ):
            raise ValueError("normalized_end must be >= normalized_start")
        return self


class Mention(FrozenModel):
    id: str
    evidence_span_id: str
    entity_type: EntityType
    text: str
    normalized_text: str
    extractor_name: str
    extractor_version: str
    score: float = Field(ge=0.0, le=1.0)
    accepted: bool = True


class TemporalExpression(FrozenModel):
    original_text: str
    start: str | None = None
    end: str | None = None
    precision: TemporalPrecision = TemporalPrecision.UNRESOLVED
    timezone: str | None = None
    is_relative: bool = False
    resolved: bool = False
    evidence_span_id: str | None = None


class ClaimParticipant(FrozenModel):
    role: str
    mention_id: str | None = None
    entity_id: str | None = None
    literal: str | None = None

    @model_validator(mode="after")
    def require_reference(self) -> ClaimParticipant:
        if not any((self.mention_id, self.entity_id, self.literal)):
            raise ValueError("participant must reference a mention, entity, or literal")
        return self


class Claim(FrozenModel):
    id: str
    kind: ClaimKind
    predicate: str
    participants: tuple[ClaimParticipant, ...]
    evidence_span_ids: tuple[str, ...]
    polarity: Polarity = Polarity.UNKNOWN
    modality: Modality = Modality.UNKNOWN
    temporal: tuple[TemporalExpression, ...] = ()
    extractor_name: str
    extractor_version: str
    score: float = Field(ge=0.0, le=1.0)
    accepted: bool = False

    @field_validator("evidence_span_ids")
    @classmethod
    def require_evidence(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("claim requires evidence")
        return value

    @field_validator("participants")
    @classmethod
    def require_participants(
        cls, value: tuple[ClaimParticipant, ...]
    ) -> tuple[ClaimParticipant, ...]:
        if not value:
            raise ValueError("claim requires at least one participant")
        return value


class Entity(FrozenModel):
    id: str
    type: EntityType
    canonical_name: str
    aliases: tuple[str, ...]
    mention_ids: tuple[str, ...]
    score: float = Field(ge=0.0, le=1.0)

    @field_validator("mention_ids")
    @classmethod
    def require_mentions(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("entity requires at least one mention")
        return value


class Event(FrozenModel):
    id: str
    type: EventType
    title: str
    claim_ids: tuple[str, ...]
    participant_entity_ids: tuple[str, ...] = ()
    temporal: tuple[TemporalExpression, ...] = ()
    score: float = Field(ge=0.0, le=1.0)
    provisional: bool = True

    @field_validator("claim_ids")
    @classmethod
    def require_claims(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("event requires at least one claim")
        return value


class Relationship(FrozenModel):
    id: str
    source_id: str
    target_id: str
    type: str
    supporting_claim_ids: tuple[str, ...]
    opposing_claim_ids: tuple[str, ...] = ()
    score: float = Field(ge=0.0, le=1.0)

    @field_validator("supporting_claim_ids")
    @classmethod
    def require_claims(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("relationship requires at least one supporting claim")
        return value


class ReviewItem(FrozenModel):
    id: str
    kind: str
    target_ids: tuple[str, ...]
    reason: str
    score: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence_span_ids: tuple[str, ...] = ()


class ReviewDecision(FrozenModel):
    id: str
    action: ReviewAction
    target_ids: tuple[str, ...]
    payload: Mapping[str, Any] = Field(default_factory=dict)
    reason: str | None = None
    supersedes: str | None = None

    @field_validator("payload", mode="after")
    @classmethod
    def freeze_payload(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        return freeze_mapping(value)

    @field_serializer("payload")
    def serialize_payload(self, value: Mapping[str, Any]) -> dict[str, Any]:
        return thaw_mapping(value)


class DocumentFailure(FrozenModel):
    code: DocumentFailureCode
    message: str
    source_reference_id: str | None = None
    stage: BuildStage
    exception_type: str | None = None


class DocumentBuildResult(FrozenModel):
    source_reference_id: str
    document_id: str | None = None
    completed_stage: BuildStage | None = None
    failures: tuple[DocumentFailure, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def succeeded(self) -> bool:
        return not self.failures


class BuildManifest(FrozenModel):
    schema_version: str = "1.0"
    docchrono_version: str
    python_version: str
    platform: str
    dependency_versions: Mapping[str, str] = Field(default_factory=dict)
    adapter_versions: Mapping[str, str] = Field(default_factory=dict)
    model_versions: Mapping[str, str] = Field(default_factory=dict)
    config_fingerprint: str

    @field_validator("dependency_versions", "adapter_versions", "model_versions", mode="after")
    @classmethod
    def freeze_versions(cls, value: Mapping[str, str]) -> Mapping[str, str]:
        return freeze_mapping(value)

    @field_serializer("dependency_versions", "adapter_versions", "model_versions")
    def serialize_versions(self, value: Mapping[str, str]) -> dict[str, Any]:
        return thaw_mapping(value)


class BuildReport(FrozenModel):
    requested_stage: BuildStage
    completed_stage: BuildStage | None
    documents: tuple[DocumentBuildResult, ...] = ()
    failures: tuple[DocumentFailure, ...] = ()
    warnings: tuple[str, ...] = ()
    review_items: tuple[ReviewItem, ...] = ()
    manifest: BuildManifest

    @property
    def complete(self) -> bool:
        return (
            self.completed_stage == self.requested_stage
            and not self.failures
            and all(result.succeeded for result in self.documents)
        )


class CaseConfig(FrozenModel):
    auto_accept_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    review_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    auto_merge_threshold: float = Field(default=0.98, ge=0.0, le=1.0)
    # Conservative alpha defaults for processing untrusted local collections.
    max_file_bytes: int = Field(default=25 * 1024 * 1024, ge=1)
    max_files: int = Field(default=10_000, ge=1)
    max_depth: int = Field(default=64, ge=1)
    max_total_bytes: int = Field(default=1024 * 1024 * 1024, ge=1)
    max_extracted_chars: int = Field(default=1_000_000, ge=1)
    max_pdf_pages: int = Field(default=2_000, ge=1)
    max_pdf_word_boxes: int = Field(default=250_000, ge=1)
    max_docx_members: int = Field(default=10_000, ge=1)
    max_docx_uncompressed_bytes: int = Field(default=64 * 1024 * 1024, ge=1)
    max_docx_member_bytes: int = Field(default=32 * 1024 * 1024, ge=1)
    max_docx_compression_ratio: float = Field(default=100.0, ge=1.0)
    max_resolution_candidates: int = Field(default=100_000, ge=1)
    max_resolution_block_size: int = Field(default=500, ge=1)
    follow_symlinks: bool = False
    default_timezone: str | None = None
    language: Literal["en"] = "en"

    @model_validator(mode="after")
    def validate_thresholds(self) -> CaseConfig:
        if self.review_threshold > self.auto_accept_threshold:
            raise ValueError("review_threshold must not exceed auto_accept_threshold")
        return self


class CaseData(FrozenModel):
    schema_version: str = "1.0"
    config: CaseConfig
    source_references: tuple[SourceReference, ...] = ()
    documents: tuple[Document, ...] = ()
    evidence_spans: tuple[EvidenceSpan, ...] = ()
    mentions: tuple[Mention, ...] = ()
    claims: tuple[Claim, ...] = ()
    entities: tuple[Entity, ...] = ()
    events: tuple[Event, ...] = ()
    relationships: tuple[Relationship, ...] = ()
    review_items: tuple[ReviewItem, ...] = ()
    review_decisions: tuple[ReviewDecision, ...] = ()
    report: BuildReport
