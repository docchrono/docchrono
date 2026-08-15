from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations

from rapidfuzz import fuzz

from docchrono.domain import (
    CaseConfig,
    Claim,
    ClaimKind,
    ClaimParticipant,
    Entity,
    EntityType,
    Event,
    EventType,
    Mention,
    Modality,
    Polarity,
    Relationship,
    ReviewItem,
)
from docchrono.domain.ids import stable_id
from docchrono.errors import IntegrityError, ResourceLimitError
from docchrono.extraction.contracts import (
    ExtractionResult,
    ResolutionCandidate,
    ResolutionResult,
)

_STRUCTURED_IDENTIFIER = re.compile(
    r"^(?P<scheme>email|case|contract|invoice|payment|account|claim|order):(?P<value>.+)$",
    re.I,
)
_HONORIFIC = re.compile(r"^(?:mr|mrs|ms|dr|prof)\s+", re.I)
_CORPORATE_SUFFIX = re.compile(
    r"\s+(?:inc|llc|ltd|corp|corporation|company|co|bank|university)$", re.I
)
_NON_ALNUM = re.compile(r"[^\w]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class _PairScore:
    score: float
    reasons: tuple[str, ...]


class _UnionFind:
    def __init__(self, values: list[str]) -> None:
        self._parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self._parent[value]
        if parent != value:
            self._parent[value] = self.find(parent)
        return self._parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        # Content IDs have a stable lexicographic ordering; never let traversal
        # order influence the representative.
        first, second = sorted((left_root, right_root))
        self._parent[second] = first


class EntityResolver:
    """Conservative, explainable entity resolution and derived views."""

    name = "docchrono.entity-resolver"
    version = "1"

    def resolve(
        self,
        extraction: ExtractionResult,
        config: CaseConfig | None = None,
    ) -> ResolutionResult:
        effective_config = config or CaseConfig()
        all_mentions = sorted(extraction.mentions, key=lambda item: item.id)
        mentions = [mention for mention in all_mentions if mention.accepted]
        evidence_document = {span.id: span.document_id for span in extraction.evidence_spans}
        self._validate_claim_mentions(extraction.claims, {item.id for item in all_mentions})

        union = _UnionFind([mention.id for mention in mentions])
        candidates: list[ResolutionCandidate] = []
        review_items: list[ReviewItem] = []
        by_type: dict[EntityType, list[Mention]] = defaultdict(list)
        for mention in mentions:
            by_type[mention.entity_type].append(mention)

        for entity_type in sorted(by_type, key=lambda item: item.value):
            for left, right in self._candidate_pairs(
                by_type[entity_type],
                max_block_size=effective_config.max_resolution_block_size,
                max_candidates=effective_config.max_resolution_candidates,
            ):
                pair = self._compare(
                    left,
                    right,
                    same_document=(
                        evidence_document.get(left.evidence_span_id)
                        == evidence_document.get(right.evidence_span_id)
                    ),
                )
                auto_merged = pair.score >= effective_config.auto_merge_threshold
                if auto_merged:
                    union.union(left.id, right.id)
                candidates.append(
                    ResolutionCandidate(
                        left_mention_id=left.id,
                        right_mention_id=right.id,
                        score=pair.score,
                        reasons=pair.reasons,
                        auto_merged=auto_merged,
                    )
                )
                if (
                    effective_config.review_threshold
                    <= pair.score
                    < effective_config.auto_merge_threshold
                ):
                    review_items.append(
                        ReviewItem(
                            id=stable_id("review", "entity-pair", left.id, right.id, pair.score),
                            kind="entity_merge_candidate",
                            target_ids=(left.id, right.id),
                            reason="; ".join(pair.reasons) or "fuzzy name similarity",
                            score=pair.score,
                            evidence_span_ids=tuple(
                                sorted((left.evidence_span_id, right.evidence_span_id))
                            ),
                        )
                    )

        groups: dict[str, list[Mention]] = defaultdict(list)
        for mention in mentions:
            groups[union.find(mention.id)].append(mention)

        entities: list[Entity] = []
        mention_to_entity: dict[str, str] = {}
        for root in sorted(groups):
            group = sorted(groups[root], key=lambda item: item.id)
            entity_id = stable_id("entity", tuple(item.id for item in group))
            canonical = self._canonical_name(group)
            aliases = tuple(
                sorted(
                    {item.text.strip() for item in group if item.text.strip()},
                    key=lambda item: (item.casefold(), item),
                )
            )
            entity = Entity(
                id=entity_id,
                type=group[0].entity_type,
                canonical_name=canonical,
                aliases=aliases,
                mention_ids=tuple(item.id for item in group),
                score=sum(item.score for item in group) / len(group),
            )
            entities.append(entity)
            mention_to_entity.update({item.id: entity_id for item in group})

        resolved_claims = tuple(
            self._resolve_claim(claim, mention_to_entity)
            for claim in sorted(extraction.claims, key=lambda item: item.id)
        )
        events, relationships = self.derive_views(resolved_claims)
        return ResolutionResult(
            entities=tuple(sorted(entities, key=lambda item: item.id)),
            claims=resolved_claims,
            events=events,
            relationships=relationships,
            review_items=tuple(sorted(review_items, key=lambda item: item.id)),
            candidates=tuple(
                sorted(
                    candidates,
                    key=lambda item: (item.left_mention_id, item.right_mention_id),
                )
            ),
            mention_to_entity=tuple(sorted(mention_to_entity.items())),
        )

    @classmethod
    def derive_views(
        cls, claims: tuple[Claim, ...]
    ) -> tuple[tuple[Event, ...], tuple[Relationship, ...]]:
        """Rebuild every claim-derived view without changing entity resolution."""

        events = cls._events(claims)
        return events, cls._relationships(claims, events)

    @classmethod
    def _compare(cls, left: Mention, right: Mention, *, same_document: bool) -> _PairScore:
        left_identifier = _STRUCTURED_IDENTIFIER.fullmatch(left.normalized_text)
        right_identifier = _STRUCTURED_IDENTIFIER.fullmatch(right.normalized_text)
        if left_identifier or right_identifier:
            if not (left_identifier and right_identifier):
                return _PairScore(0.0, ("identifier/name forms are not directly comparable",))
            if (
                left_identifier.group("scheme").casefold()
                == right_identifier.group("scheme").casefold()
                and left_identifier.group("value").casefold()
                == right_identifier.group("value").casefold()
            ):
                return _PairScore(1.0, ("exact durable identifier",))
            return _PairScore(0.0, ("different durable identifiers",))

        left_name = cls._normalize_name(left.normalized_text)
        right_name = cls._normalize_name(right.normalized_text)
        if not left_name or not right_name:
            return _PairScore(0.0, ("empty normalized alias",))
        if left_name == right_name:
            return _PairScore(0.985, ("exact normalized alias",))

        left_aliases = cls._aliases(left_name, left.entity_type)
        right_aliases = cls._aliases(right_name, right.entity_type)
        common = left_aliases.intersection(right_aliases)
        if common:
            return _PairScore(0.99, ("exact conservative alias or acronym",))

        similarity = fuzz.token_sort_ratio(left_name, right_name) / 100.0
        # Fuzzy similarity alone never reaches the default automatic threshold.
        score = min(0.97, similarity * 0.97)
        reasons = [f"RapidFuzz token similarity {similarity:.3f}"]
        if same_document and score > 0:
            score = min(0.97, score + 0.01)
            reasons.append("same-document context supports comparison")
        return _PairScore(round(score, 6), tuple(reasons))

    @classmethod
    def _candidate_pairs(
        cls,
        mentions: list[Mention],
        *,
        max_block_size: int,
        max_candidates: int,
    ) -> tuple[tuple[Mention, Mention], ...]:
        """Return deterministic, bounded pairs that share a conservative blocking key."""

        mention_by_id = {mention.id: mention for mention in mentions}
        blocks: dict[str, list[str]] = defaultdict(list)
        for mention in mentions:
            for key in cls._blocking_keys(mention):
                blocks[key].append(mention.id)

        pair_ids: set[tuple[str, str]] = set()
        for key in sorted(blocks):
            values = sorted(set(blocks[key]))
            if len(values) > max_block_size:
                raise ResourceLimitError(
                    f"entity-resolution block {key!r} contains {len(values)} mentions; "
                    f"limit is {max_block_size}"
                )
            for left_id, right_id in combinations(values, 2):
                pair_ids.add((left_id, right_id))
                if len(pair_ids) > max_candidates:
                    raise ResourceLimitError(
                        "entity-resolution candidate budget exceeded: "
                        f"more than {max_candidates} pairs"
                    )
        return tuple(
            (mention_by_id[left_id], mention_by_id[right_id])
            for left_id, right_id in sorted(pair_ids)
        )

    @classmethod
    def _blocking_keys(cls, mention: Mention) -> tuple[str, ...]:
        identifier = _STRUCTURED_IDENTIFIER.fullmatch(mention.normalized_text)
        if identifier is not None:
            # Durable identifiers need only be compared with the same normalized
            # identifier. Comparing every email/account pair is both useless and
            # attacker-controlled quadratic work.
            return (f"identifier:{mention.normalized_text.casefold()}",)

        normalized = cls._normalize_name(mention.normalized_text)
        if not normalized:
            return ()
        keys = {f"exact:{normalized}"}
        if mention.entity_type == EntityType.ORGANIZATION:
            aliases = cls._aliases(normalized, mention.entity_type)
            keys.update(f"alias:{alias.casefold()}" for alias in aliases)
        elif mention.entity_type in {EntityType.PERSON, EntityType.LOCATION}:
            words = normalized.split()
            if len(words) >= 2:
                keys.add(f"last-token:{words[-1]}")
        return tuple(sorted(keys))

    @staticmethod
    def _normalize_name(value: str) -> str:
        normalized = unicodedata.normalize("NFKC", value).casefold().strip()
        normalized = _HONORIFIC.sub("", normalized)
        return " ".join(_NON_ALNUM.sub(" ", normalized).split())

    @classmethod
    def _aliases(cls, normalized: str, entity_type: EntityType) -> set[str]:
        aliases = {normalized}
        if entity_type == EntityType.ORGANIZATION:
            core = _CORPORATE_SUFFIX.sub("", normalized).strip()
            if core:
                aliases.add(core)
            words = [word for word in core.split() if word not in {"the", "of", "and"}]
            if len(words) >= 2:
                aliases.add("".join(word[0] for word in words).casefold())
        return aliases

    @staticmethod
    def _canonical_name(group: list[Mention]) -> str:
        non_identifiers = [
            item for item in group if _STRUCTURED_IDENTIFIER.fullmatch(item.normalized_text) is None
        ]
        candidates = non_identifiers or group
        strongest = max(item.score for item in candidates)
        supported = [item for item in candidates if item.score >= strongest - 0.05]
        return min(
            supported,
            key=lambda item: (
                -len(item.text.split()),
                -len(item.text),
                item.text.casefold(),
                item.text,
            ),
        ).text.strip()

    @staticmethod
    def _validate_claim_mentions(claims: tuple[Claim, ...], mention_ids: set[str]) -> None:
        for claim in claims:
            for participant in claim.participants:
                if participant.mention_id is not None and participant.mention_id not in mention_ids:
                    raise IntegrityError(
                        f"claim {claim.id} references unknown mention {participant.mention_id}"
                    )

    @staticmethod
    def _resolve_claim(claim: Claim, mapping: dict[str, str]) -> Claim:
        participants = tuple(
            ClaimParticipant(
                role=participant.role,
                mention_id=participant.mention_id,
                entity_id=(
                    mapping.get(participant.mention_id)
                    if participant.mention_id is not None
                    else participant.entity_id
                ),
                literal=participant.literal,
            )
            for participant in claim.participants
        )
        return claim.model_copy(update={"participants": participants})

    @classmethod
    def _events(cls, claims: tuple[Claim, ...]) -> tuple[Event, ...]:
        events: list[Event] = []
        for claim in claims:
            if (
                claim.kind != ClaimKind.EVENT
                or not claim.accepted
                or claim.polarity != Polarity.AFFIRMED
                or claim.modality != Modality.ASSERTED
            ):
                continue
            participant_ids = tuple(
                sorted(
                    {
                        participant.entity_id
                        for participant in claim.participants
                        if participant.entity_id is not None
                    }
                )
            )
            literal = next(
                (
                    participant.literal
                    for participant in claim.participants
                    if participant.literal is not None
                ),
                None,
            )
            title = claim.predicate.replace("_", " ").title()
            if literal:
                title = f"{title}: {literal}"
            events.append(
                Event(
                    id=stable_id("event", claim.id),
                    type=cls._event_type(claim.predicate),
                    title=title,
                    claim_ids=(claim.id,),
                    participant_entity_ids=participant_ids,
                    temporal=claim.temporal,
                    score=claim.score,
                    provisional=True,
                )
            )
        return tuple(sorted(events, key=lambda item: item.id))

    @staticmethod
    def _event_type(predicate: str) -> EventType:
        normalized = predicate.upper()
        if normalized in {"APPROVED", "REJECTED", "AUTHORIZED"}:
            return EventType.DECISION
        if normalized in {"PAID", "TRANSFERRED", "PURCHASED", "SOLD"}:
            return EventType.TRANSACTION
        if normalized in {"EMAILED", "NOTIFIED", "EMAIL_SENT"}:
            return EventType.COMMUNICATION
        if normalized == "MET":
            return EventType.MEETING
        if normalized in {"SIGNED", "EXECUTED", "FILED", "SUBMITTED"}:
            return EventType.ACTION
        return EventType.OTHER

    @staticmethod
    def _relationships(
        claims: tuple[Claim, ...], events: tuple[Event, ...]
    ) -> tuple[Relationship, ...]:
        supporting: dict[tuple[str, str, str], list[Claim]] = defaultdict(list)
        opposing: dict[tuple[str, str, str], list[Claim]] = defaultdict(list)
        for claim in claims:
            if claim.kind != ClaimKind.RELATIONSHIP or not claim.accepted:
                continue
            if claim.modality != Modality.ASSERTED:
                continue
            source = next(
                (
                    participant.entity_id
                    for participant in claim.participants
                    if participant.role == "source" and participant.entity_id is not None
                ),
                None,
            )
            target = next(
                (
                    participant.entity_id
                    for participant in claim.participants
                    if participant.role == "target" and participant.entity_id is not None
                ),
                None,
            )
            if source is None or target is None:
                continue
            key = (source, target, claim.predicate)
            if claim.polarity == Polarity.AFFIRMED:
                supporting[key].append(claim)
            elif claim.polarity == Polarity.NEGATED:
                opposing[key].append(claim)

        relationships = [
            Relationship(
                id=stable_id("relationship", source, target, predicate),
                source_id=source,
                target_id=target,
                type=predicate,
                supporting_claim_ids=tuple(sorted(claim.id for claim in support_claims)),
                opposing_claim_ids=tuple(
                    sorted(claim.id for claim in opposing.get((source, target, predicate), ()))
                ),
                score=max(claim.score for claim in support_claims),
            )
            for (source, target, predicate), support_claims in supporting.items()
        ]
        relationships.extend(
            Relationship(
                id=stable_id("relationship", entity_id, event.id, "PARTICIPATED_IN"),
                source_id=entity_id,
                target_id=event.id,
                type="PARTICIPATED_IN",
                supporting_claim_ids=event.claim_ids,
                score=event.score,
            )
            for event in events
            for entity_id in event.participant_entity_ids
        )
        return tuple(sorted(relationships, key=lambda item: item.id))
