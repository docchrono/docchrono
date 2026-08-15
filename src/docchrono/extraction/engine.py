from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol, TypeVar, cast

from docchrono.domain import (
    BoundingBox,
    CaseConfig,
    Claim,
    ClaimParticipant,
    Document,
    DocumentSegment,
    EvidenceSpan,
    Mention,
    ReviewItem,
    TemporalExpression,
)
from docchrono.domain.ids import stable_id
from docchrono.errors import AdapterContractError
from docchrono.extraction.contracts import (
    CandidateParticipant,
    ClaimCandidate,
    ExtractionBatch,
    ExtractionContext,
    ExtractionResult,
    Extractor,
    MentionCandidate,
    TemporalCandidate,
)
from docchrono.extraction.nlp import EnglishNlpPipeline
from docchrono.extraction.registry import ExtractorRegistry
from docchrono.extraction.standard import StandardEnglishExtractor


@dataclass(frozen=True, slots=True)
class _AdapterBatch:
    document: Document
    extractor: Extractor
    batch: ExtractionBatch


@dataclass(frozen=True, slots=True)
class _MentionOrigin:
    document: Document
    extractor: Extractor
    candidate: MentionCandidate


class _Identified(Protocol):
    id: str


T = TypeVar("T", bound=_Identified)


class ExtractionEngine:
    """Validate adapters and produce deterministic evidence-linked findings."""

    def __init__(
        self,
        registry: ExtractorRegistry | None = None,
        *,
        nlp: EnglishNlpPipeline | None = None,
        model: str | object | None = None,
    ) -> None:
        if nlp is not None and model is not None:
            raise ValueError("pass either nlp or model, not both")
        if nlp is not None:
            self._nlp = nlp
        else:
            # EnglishNlpPipeline performs the runtime Language check. Keeping the
            # loose public type here avoids importing spaCy types into callers.
            self._nlp = EnglishNlpPipeline(model=model)  # type: ignore[arg-type]
        self.registry = registry or ExtractorRegistry((StandardEnglishExtractor(),))

    @classmethod
    def standard(
        cls,
        *,
        model: str | object | None = None,
    ) -> ExtractionEngine:
        return cls(model=model)

    def with_extractor(self, extractor: Extractor, *, replace: bool = False) -> ExtractionEngine:
        self.registry.register(extractor, replace=replace)
        return self

    @property
    def model_versions(self) -> dict[str, str]:
        identity = self._nlp.identity
        suffix = "+fallback" if identity.fallback else ""
        return {identity.name: f"{identity.version}{suffix}"}

    def extract(
        self,
        documents: tuple[Document, ...] | list[Document],
        config: CaseConfig | None = None,
    ) -> ExtractionResult:
        effective_config = config or CaseConfig()
        context = ExtractionContext(config=effective_config, nlp=self._nlp)
        adapter_batches: list[_AdapterBatch] = []
        for document in sorted(documents, key=lambda item: item.id):
            self._validate_document(document)
            for extractor in self.registry.extractors:
                batch = extractor.extract(document, context)
                self._validate_batch(document, extractor, batch)
                adapter_batches.append(_AdapterBatch(document, extractor, batch))

        evidence: dict[str, EvidenceSpan] = {}
        mention_origins = [
            _MentionOrigin(item.document, item.extractor, candidate)
            for item in adapter_batches
            for candidate in item.batch.mentions
        ]
        selected_origins, mention_key_map = self._select_mentions(mention_origins)
        mentions: list[Mention] = []
        mention_by_signature: dict[tuple[object, ...], Mention] = {}
        for origin in selected_origins:
            span = self._evidence(
                origin.document, origin.candidate.normalized_start, origin.candidate.normalized_end
            )
            evidence[span.id] = span
            candidate = origin.candidate
            normalized_text = candidate.normalized_text or " ".join(
                origin.document.normalized_text[
                    candidate.normalized_start : candidate.normalized_end
                ]
                .casefold()
                .split()
            )
            mention = Mention(
                id=stable_id(
                    "mention",
                    span.id,
                    candidate.entity_type,
                    normalized_text,
                    origin.extractor.name,
                    origin.extractor.version,
                ),
                evidence_span_id=span.id,
                entity_type=candidate.entity_type,
                text=origin.document.normalized_text[
                    candidate.normalized_start : candidate.normalized_end
                ],
                normalized_text=normalized_text,
                extractor_name=origin.extractor.name,
                extractor_version=origin.extractor.version,
                score=candidate.score,
                accepted=candidate.score >= effective_config.auto_accept_threshold,
            )
            signature = self._mention_signature(origin)
            mention_by_signature[signature] = mention
            mentions.append(mention)

        # Map every original adapter-local key, including candidates suppressed as
        # exact duplicates, onto the selected canonical mention.
        mention_ids_by_key: dict[tuple[str, str, str], str] = {}
        for origin, selected_signature in mention_key_map:
            mention_ids_by_key[
                (origin.document.id, origin.extractor.name.casefold(), origin.candidate.key)
            ] = mention_by_signature[selected_signature].id

        temporals: list[TemporalExpression] = []
        temporal_maps: dict[tuple[str, str], tuple[TemporalExpression, ...]] = {}
        for item in adapter_batches:
            converted: list[TemporalExpression] = []
            for candidate in item.batch.temporals:
                span = self._evidence(
                    item.document, candidate.normalized_start, candidate.normalized_end
                )
                evidence[span.id] = span
                expression = candidate.expression.model_copy(update={"evidence_span_id": span.id})
                converted.append(expression)
                temporals.append(expression)
            temporal_maps[(item.document.id, item.extractor.name.casefold())] = tuple(converted)

        claims: list[Claim] = []
        for item in adapter_batches:
            key = (item.document.id, item.extractor.name.casefold())
            adapter_temporals = temporal_maps[key]
            for candidate in item.batch.claims:
                span = self._evidence(
                    item.document, candidate.normalized_start, candidate.normalized_end
                )
                evidence[span.id] = span
                participants = tuple(
                    ClaimParticipant(
                        role=participant.role,
                        mention_id=(
                            mention_ids_by_key[
                                (
                                    item.document.id,
                                    item.extractor.name.casefold(),
                                    participant.mention_key,
                                )
                            ]
                            if participant.mention_key is not None
                            else None
                        ),
                        literal=participant.literal,
                    )
                    for participant in candidate.participants
                )
                temporal = tuple(adapter_temporals[index] for index in candidate.temporal_keys)
                accepted = candidate.score >= effective_config.auto_accept_threshold
                claims.append(
                    Claim(
                        id=stable_id(
                            "claim",
                            item.document.id,
                            span.id,
                            candidate.kind,
                            candidate.predicate,
                            participants,
                            candidate.polarity,
                            candidate.modality,
                            temporal,
                            item.extractor.name,
                            item.extractor.version,
                        ),
                        kind=candidate.kind,
                        predicate=candidate.predicate,
                        participants=participants,
                        evidence_span_ids=(span.id,),
                        polarity=candidate.polarity,
                        modality=candidate.modality,
                        temporal=temporal,
                        extractor_name=item.extractor.name,
                        extractor_version=item.extractor.version,
                        score=candidate.score,
                        accepted=accepted,
                    )
                )

        mentions = self._unique_records(mentions)
        claims = self._unique_records(claims)
        temporals = self._unique_temporals(temporals)
        review_items: list[ReviewItem] = []
        accepted_mentions: list[str] = []
        review_mentions: list[str] = []
        deferred_mentions: list[str] = []
        for mention in mentions:
            if mention.score >= effective_config.auto_accept_threshold:
                accepted_mentions.append(mention.id)
            elif mention.score >= effective_config.review_threshold:
                review_mentions.append(mention.id)
                review_items.append(self._review_for_mention(mention, deferred=False))
            else:
                deferred_mentions.append(mention.id)
                review_items.append(self._review_for_mention(mention, deferred=True))

        accepted_claims: list[str] = []
        review_claims: list[str] = []
        deferred_claims: list[str] = []
        for claim in claims:
            if claim.score >= effective_config.auto_accept_threshold:
                accepted_claims.append(claim.id)
            elif claim.score >= effective_config.review_threshold:
                review_claims.append(claim.id)
                review_items.append(self._review_for_claim(claim, deferred=False))
            else:
                deferred_claims.append(claim.id)
                review_items.append(self._review_for_claim(claim, deferred=True))

        return ExtractionResult(
            evidence_spans=tuple(sorted(evidence.values(), key=lambda item: item.id)),
            mentions=tuple(mentions),
            temporal_expressions=tuple(temporals),
            claims=tuple(claims),
            review_items=tuple(sorted(review_items, key=lambda item: item.id)),
            accepted_mention_ids=tuple(sorted(accepted_mentions)),
            review_mention_ids=tuple(sorted(review_mentions)),
            deferred_mention_ids=tuple(sorted(deferred_mentions)),
            accepted_claim_ids=tuple(sorted(accepted_claims)),
            review_claim_ids=tuple(sorted(review_claims)),
            deferred_claim_ids=tuple(sorted(deferred_claims)),
            adapter_versions=tuple(
                (extractor.name, extractor.version) for extractor in self.registry.extractors
            ),
        )

    @staticmethod
    def _validate_document(document: Document) -> None:
        mapping = document.normalized_to_raw
        if len(mapping) != len(document.normalized_text) + 1:
            raise AdapterContractError(f"invalid normalized offset map for {document.id}")
        if any(left > right for left, right in pairwise(mapping)):
            raise AdapterContractError(f"non-monotonic normalized offset map for {document.id}")
        if mapping and (mapping[0] < 0 or mapping[-1] > len(document.raw_text)):
            raise AdapterContractError(f"normalized offset map escapes raw text for {document.id}")

    @classmethod
    def _validate_batch(
        cls,
        document: Document,
        extractor: Extractor,
        batch: object,
    ) -> None:
        if not isinstance(batch, ExtractionBatch):
            raise AdapterContractError(
                f"{extractor.name} must return ExtractionBatch, got {type(batch).__name__}"
            )
        mention_keys: set[str] = set()
        for candidate in cast(tuple[object, ...], batch.mentions):
            if not isinstance(candidate, MentionCandidate):
                raise AdapterContractError(f"{extractor.name} emitted a non-MentionCandidate value")
            cls._validate_candidate_span(document, extractor, candidate)
            if not candidate.key or candidate.key in mention_keys:
                raise AdapterContractError(
                    f"{extractor.name} emitted a duplicate/empty mention key"
                )
            mention_keys.add(candidate.key)
        for candidate in cast(tuple[object, ...], batch.temporals):
            if not isinstance(candidate, TemporalCandidate):
                raise AdapterContractError(
                    f"{extractor.name} emitted a non-TemporalCandidate value"
                )
            cls._validate_candidate_span(document, extractor, candidate)
            quote = document.normalized_text[candidate.normalized_start : candidate.normalized_end]
            if quote != candidate.expression.original_text:
                raise AdapterContractError(
                    f"{extractor.name} temporal original_text does not match its span"
                )
        claim_keys: set[str] = set()
        for candidate in cast(tuple[object, ...], batch.claims):
            if not isinstance(candidate, ClaimCandidate):
                raise AdapterContractError(f"{extractor.name} emitted a non-ClaimCandidate value")
            cls._validate_candidate_span(document, extractor, candidate)
            if not candidate.key or candidate.key in claim_keys:
                raise AdapterContractError(f"{extractor.name} emitted a duplicate/empty claim key")
            claim_keys.add(candidate.key)
            if not candidate.predicate.strip() or not candidate.participants:
                raise AdapterContractError(f"{extractor.name} emitted an incomplete claim")
            for participant in cast(tuple[object, ...], candidate.participants):
                if not isinstance(participant, CandidateParticipant):
                    raise AdapterContractError(
                        f"{extractor.name} emitted a non-CandidateParticipant value"
                    )
                count = int(participant.mention_key is not None) + int(
                    participant.literal is not None
                )
                if (
                    count != 1
                    or not participant.role.strip()
                    or (participant.literal is not None and not participant.literal.strip())
                ):
                    raise AdapterContractError(
                        f"{extractor.name} emitted an invalid claim participant"
                    )
                if (
                    participant.mention_key is not None
                    and participant.mention_key not in mention_keys
                ):
                    raise AdapterContractError(
                        f"{extractor.name} claim references unknown mention key "
                        f"{participant.mention_key!r}"
                    )
            if any(index < 0 or index >= len(batch.temporals) for index in candidate.temporal_keys):
                raise AdapterContractError(
                    f"{extractor.name} claim references an unknown temporal candidate"
                )

    @staticmethod
    def _validate_candidate_span(
        document: Document,
        extractor: Extractor,
        candidate: MentionCandidate | TemporalCandidate | ClaimCandidate,
    ) -> None:
        if not math.isfinite(candidate.score) or not 0.0 <= candidate.score <= 1.0:
            raise AdapterContractError(f"{extractor.name} emitted an invalid score")
        start = candidate.normalized_start
        end = candidate.normalized_end
        if start < 0 or end <= start or end > len(document.normalized_text):
            raise AdapterContractError(
                f"{extractor.name} emitted a span outside document {document.id}"
            )
        mapping = document.normalized_to_raw
        if mapping[start] > mapping[end] or mapping[end] > len(document.raw_text):
            raise AdapterContractError(
                f"{extractor.name} emitted a span with invalid raw provenance"
            )

    @classmethod
    def _select_mentions(
        cls,
        origins: list[_MentionOrigin],
    ) -> tuple[list[_MentionOrigin], list[tuple[_MentionOrigin, tuple[object, ...]]]]:
        grouped: dict[tuple[object, ...], list[_MentionOrigin]] = {}
        for origin in origins:
            grouped.setdefault(cls._mention_signature(origin), []).append(origin)
        selected: list[_MentionOrigin] = []
        mapping: list[tuple[_MentionOrigin, tuple[object, ...]]] = []
        for signature, values in sorted(grouped.items(), key=lambda item: repr(item[0])):
            winner = min(
                values,
                key=lambda item: (
                    -item.candidate.score,
                    item.extractor.order,
                    item.extractor.name.casefold(),
                    item.extractor.version,
                    item.candidate.key,
                ),
            )
            selected.append(winner)
            mapping.extend((item, signature) for item in values)
        selected.sort(
            key=lambda item: (
                item.document.id,
                item.candidate.normalized_start,
                item.candidate.normalized_end,
                item.candidate.entity_type.value,
                item.extractor.name.casefold(),
            )
        )
        return selected, mapping

    @staticmethod
    def _mention_signature(origin: _MentionOrigin) -> tuple[object, ...]:
        candidate = origin.candidate
        normalized = candidate.normalized_text or " ".join(
            origin.document.normalized_text[candidate.normalized_start : candidate.normalized_end]
            .casefold()
            .split()
        )
        return (
            origin.document.id,
            candidate.normalized_start,
            candidate.normalized_end,
            candidate.entity_type,
            normalized,
        )

    @staticmethod
    def _evidence(document: Document, start: int, end: int) -> EvidenceSpan:
        raw_start = document.normalized_to_raw[start]
        raw_end = document.normalized_to_raw[end]
        segment = ExtractionEngine._best_segment(document.segments, raw_start, raw_end)
        quote = document.raw_text[raw_start:raw_end]
        return EvidenceSpan(
            id=stable_id("evidence", document.id, raw_start, raw_end, quote),
            document_id=document.id,
            raw_start=raw_start,
            raw_end=raw_end,
            normalized_start=start,
            normalized_end=end,
            quote=quote,
            page=segment.page if segment else None,
            paragraph=segment.paragraph if segment else None,
            sentence=segment.sentence if segment else None,
            field=segment.field if segment else None,
            boxes=ExtractionEngine._evidence_boxes(document, raw_start, raw_end),
        )

    @staticmethod
    def _evidence_boxes(
        document: Document, raw_start: int, raw_end: int
    ) -> tuple[BoundingBox, ...]:
        raw_boxes = document.metadata.get("word_boxes", ())
        if not isinstance(raw_boxes, (tuple, list)):
            return ()
        values = cast("tuple[object, ...] | list[object]", raw_boxes)
        selected: list[tuple[int, BoundingBox]] = []
        for value in values:
            if not isinstance(value, Mapping):
                continue
            item = cast("Mapping[str, object]", value)
            box_start = item.get("raw_start")
            box_end = item.get("raw_end")
            if (
                not isinstance(box_start, int)
                or isinstance(box_start, bool)
                or not isinstance(box_end, int)
                or isinstance(box_end, bool)
                or box_end <= raw_start
                or box_start >= raw_end
            ):
                continue
            try:
                box = BoundingBox.model_validate(
                    {
                        "page": item.get("page"),
                        "x0": item.get("x0"),
                        "y0": item.get("y0"),
                        "x1": item.get("x1"),
                        "y1": item.get("y1"),
                    }
                )
            except (TypeError, ValueError):
                continue
            selected.append((box_start, box))
        ordered = sorted(selected, key=lambda value: (value[0], value[1].page))
        return tuple(box for _, box in ordered)

    @staticmethod
    def _best_segment(
        segments: tuple[DocumentSegment, ...], raw_start: int, raw_end: int
    ) -> DocumentSegment | None:
        containing = [
            segment
            for segment in segments
            if segment.raw_start <= raw_start and segment.raw_end >= raw_end
        ]
        if not containing:
            return None
        return min(
            containing,
            key=lambda item: (
                item.raw_end - item.raw_start,
                item.kind,
                item.raw_start,
                item.raw_end,
            ),
        )

    @staticmethod
    def _unique_records(records: list[T]) -> list[T]:
        by_id = {item.id: item for item in records}
        return [by_id[key] for key in sorted(by_id)]

    @staticmethod
    def _unique_temporals(values: list[TemporalExpression]) -> list[TemporalExpression]:
        selected: dict[str, TemporalExpression] = {}
        for value in values:
            key = stable_id("temporal", value)
            selected[key] = value
        return [selected[key] for key in sorted(selected)]

    @staticmethod
    def _review_for_mention(mention: Mention, *, deferred: bool) -> ReviewItem:
        reason = (
            "candidate retained below the review threshold"
            if deferred
            else "candidate requires review before acceptance"
        )
        return ReviewItem(
            id=stable_id("review", "mention", mention.id, reason),
            kind="mention_candidate_deferred" if deferred else "mention_candidate",
            target_ids=(mention.id,),
            reason=reason,
            score=mention.score,
            evidence_span_ids=(mention.evidence_span_id,),
        )

    @staticmethod
    def _review_for_claim(claim: Claim, *, deferred: bool) -> ReviewItem:
        reason = (
            "claim candidate retained below the review threshold"
            if deferred
            else "claim candidate requires review before acceptance"
        )
        return ReviewItem(
            id=stable_id("review", "claim", claim.id, reason),
            kind="claim_candidate_deferred" if deferred else "claim_candidate",
            target_ids=(claim.id,),
            reason=reason,
            score=claim.score,
            evidence_span_ids=claim.evidence_span_ids,
        )
