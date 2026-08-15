from __future__ import annotations

import bisect
import re
from collections.abc import Iterable, Mapping
from email.utils import getaddresses
from typing import cast

from docchrono.domain import ClaimKind, Document, EntityType, Modality, Polarity
from docchrono.extraction.contracts import (
    CandidateParticipant,
    ClaimCandidate,
    ExtractionBatch,
    ExtractionContext,
    MentionCandidate,
    TemporalCandidate,
)
from docchrono.extraction.nlp import EnglishNlpPipeline
from docchrono.extraction.temporal import TemporalNormalizer

_EMAIL = re.compile(r"(?<![\w.+-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,63}(?![\w.-])", re.I)
_IDENTIFIER = re.compile(
    r"\b(?P<kind>case|contract|invoice|payment|account|claim|order)"
    r"\s*(?:#|no\.?|number)?\s*"
    r"(?P<value>(?=[A-Z0-9._/-]*\d)[A-Z0-9][A-Z0-9._/-]{2,})\b",
    re.I,
)
_TITLE_PERSON = re.compile(
    r"\b(?:Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.)\s+"
    r"[A-Z][A-Za-z'\u2019\u2013-]+(?:\s+[A-Z][A-Za-z'\u2019\u2013-]+){0,3}\b"
)
_TWO_PART_PERSON = re.compile(
    r"\b[A-Z][a-z'\u2019\u2013-]{1,30}\s+[A-Z][a-z'\u2019\u2013-]{1,30}\b"
)
_ORGANIZATION = re.compile(
    r"\b[A-Z][A-Za-z&'\u2019.-]*(?:\s+[A-Z][A-Za-z&'\u2019.-]*){0,5}\s+"
    r"(?:Inc\.?|LLC|Ltd\.?|Corp\.?|Corporation|Company|Bank|University)\b"
)
_LOCATION = re.compile(
    r"\b(?:in|at|from)\s+(?P<location>[A-Z][A-Za-z.'\u2019\u2013-]+"
    r"(?:\s+[A-Z][A-Za-z.'\u2019\u2013-]+){0,2}"
    r"(?:,\s*[A-Z][A-Za-z.'\u2019\u2013-]+)?)\b"
)

_RELATION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "WORKS_FOR",
        re.compile(
            r"\b(?P<source>[A-Z][A-Za-z'\u2019\u2013-]+"
            r"(?:\s+[A-Z][A-Za-z'\u2019\u2013-]+){1,3})\s+"
            r"(?P<qualifier>(?:(?:does|did)\s+not\s+|never\s+|may\s+|might\s+|"
            r"allegedly\s+)?)"
            r"(?:work|works|worked)\s+for\s+"
            r"(?P<target>[A-Z][A-Za-z&'\u2019.-]*"
            r"(?:\s+[A-Z][A-Za-z&'\u2019.-]*){0,5}\s+"
            r"(?:Inc\.?|LLC|Ltd\.?|Corp\.?|Corporation|Company|Bank|University))\b"
        ),
    ),
    (
        "OFFICER_OF",
        re.compile(
            r"\b(?P<source>[A-Z][A-Za-z'\u2019\u2013-]+"
            r"(?:\s+[A-Z][A-Za-z'\u2019\u2013-]+){1,3})\s+"
            r"(?P<qualifier>(?:is\s+not\s+|was\s+not\s+|may\s+be\s+|allegedly\s+)?)"
            r"(?:is|was)?\s*(?:the\s+)?(?:CEO|CFO|COO|president|director)\s+of\s+"
            r"(?P<target>[A-Z][A-Za-z&'\u2019.-]*"
            r"(?:\s+[A-Z][A-Za-z&'\u2019.-]*){0,5}\s+"
            r"(?:Inc\.?|LLC|Ltd\.?|Corp\.?|Corporation|Company|Bank|University))\b",
            re.I,
        ),
    ),
)

_EVENT_PATTERN = re.compile(
    r"\b(?P<source>[A-Z][A-Za-z&'\u2019.-]*"
    r"(?:[ \t]+[A-Z][A-Za-z&'\u2019.-]*){0,4}?)[ \t]+"
    r"(?P<qualifier>(?:(?:does|did|has|had)\s+not\s+|never\s+|may\s+|might\s+|"
    r"could\s+|allegedly\s+|reportedly\s+)?)"
    r"(?P<verb>approve(?:d)?|reject(?:ed)?|authorize(?:d)?|sign(?:ed)?|execute(?:d)?|"
    r"file(?:d)?|submit(?:ted)?|pay|paid|transfer(?:red)?|purchase(?:d)?|sell|sold|"
    r"email(?:ed)?|notify|notified)\s+"
    r"(?P<target>[^.!?\n]{1,120}?)(?=[.!?\n]|$)",
    re.I,
)

_MEETING_PATTERN = re.compile(
    r"\b(?P<source>[A-Z][A-Za-z'\u2019\u2013-]+"
    r"(?:\s+[A-Z][A-Za-z'\u2019\u2013-]+){1,3})\s+"
    r"(?P<qualifier>(?:(?:did|does)\s+not\s+|may\s+|might\s+|allegedly\s+)?)"
    r"(?:met|meets)\s+with\s+"
    r"(?P<target>[A-Z][A-Za-z'\u2019\u2013-]+"
    r"(?:\s+[A-Z][A-Za-z'\u2019\u2013-]+){1,3})\b",
    re.I,
)

_GENERIC_PERSON_STOPWORDS = {
    "Annual Report",
    "Board Meeting",
    "Case Number",
    "Contract Number",
    "Dear Sir",
    "Email Subject",
    "Invoice Number",
    "New York",  # extracted by a statistical model or location context instead
    "Payment Number",
}
_NON_PERSON_FIRST_WORDS = {
    "A",
    "An",
    "At",
    "By",
    "For",
    "From",
    "In",
    "On",
    "The",
    "To",
}


class StandardEnglishExtractor:
    """Finite, high-precision English rules plus model-provided entity spans."""

    name = "docchrono.standard-english"
    version = "1"
    order = 100

    def __init__(self, temporal: TemporalNormalizer | None = None) -> None:
        self._temporal = temporal or TemporalNormalizer()

    def extract(self, document: Document, context: ExtractionContext) -> ExtractionBatch:
        if not isinstance(context.nlp, EnglishNlpPipeline):
            raise TypeError("standard extractor requires EnglishNlpPipeline")
        mentions = self._entity_mentions(document, context.nlp)
        temporals = list(
            self._temporal.find(document, default_timezone=context.config.default_timezone)
        )
        structural_mentions, structural_claims = self._email_claims(document, mentions, temporals)
        mentions.extend(structural_mentions)
        claims = list(structural_claims)
        claims.extend(self._semantic_claims(document, mentions, temporals))
        return ExtractionBatch(
            mentions=tuple(self._dedupe_mentions(mentions)),
            temporals=tuple(temporals),
            claims=tuple(claims),
        )

    def _entity_mentions(
        self,
        document: Document,
        nlp: EnglishNlpPipeline,
    ) -> list[MentionCandidate]:
        text = document.normalized_text
        mentions: list[MentionCandidate] = []
        doc = nlp.process(text)
        label_map = {
            "PERSON": EntityType.PERSON,
            "PER": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOC": EntityType.LOCATION,
            "FAC": EntityType.LOCATION,
        }
        for index, ent in enumerate(doc.ents):
            entity_type = label_map.get(ent.label_)
            if entity_type is None:
                continue
            score = 0.92 if not nlp.fallback else 0.78
            mentions.append(
                self._mention(
                    key=f"spacy:{index}:{ent.start_char}:{ent.end_char}",
                    start=ent.start_char,
                    end=ent.end_char,
                    entity_type=entity_type,
                    score=score,
                    text=text,
                )
            )

        for index, match in enumerate(_EMAIL.finditer(text)):
            mentions.append(
                self._mention(
                    key=f"email:{index}:{match.start()}",
                    start=match.start(),
                    end=match.end(),
                    entity_type=EntityType.OTHER,
                    score=0.995,
                    text=text,
                    normalized=f"email:{match.group(0).casefold()}",
                )
            )
        for index, match in enumerate(_IDENTIFIER.finditer(text)):
            kind = match.group("kind").casefold()
            value = match.group("value").casefold()
            mentions.append(
                self._mention(
                    key=f"identifier:{index}:{match.start()}",
                    start=match.start(),
                    end=match.end(),
                    entity_type=EntityType.OTHER,
                    score=0.98,
                    text=text,
                    normalized=f"{kind}:{value}",
                )
            )

        self._append_regex_mentions(
            mentions, text, _TITLE_PERSON, EntityType.PERSON, 0.90, "person-title"
        )
        self._append_regex_mentions(
            mentions, text, _ORGANIZATION, EntityType.ORGANIZATION, 0.91, "organization-suffix"
        )
        for index, match in enumerate(_LOCATION.finditer(text)):
            start, end = match.span("location")
            mentions.append(
                self._mention(
                    key=f"location-context:{index}:{start}",
                    start=start,
                    end=end,
                    entity_type=EntityType.LOCATION,
                    score=0.72,
                    text=text,
                )
            )
        for index, match in enumerate(_TWO_PART_PERSON.finditer(text)):
            surface = match.group(0)
            overlaps_non_person = any(
                candidate.entity_type != EntityType.PERSON
                and candidate.normalized_start < match.end()
                and candidate.normalized_end > match.start()
                for candidate in mentions
            )
            if (
                surface in _GENERIC_PERSON_STOPWORDS
                or surface.split(maxsplit=1)[0] in _NON_PERSON_FIRST_WORDS
                or overlaps_non_person
            ):
                continue
            mentions.append(
                self._mention(
                    key=f"person-titlecase:{index}:{match.start()}",
                    start=match.start(),
                    end=match.end(),
                    entity_type=EntityType.PERSON,
                    score=0.70,
                    text=text,
                )
            )
        return self._dedupe_mentions(mentions)

    @staticmethod
    def _append_regex_mentions(
        output: list[MentionCandidate],
        text: str,
        pattern: re.Pattern[str],
        entity_type: EntityType,
        score: float,
        prefix: str,
    ) -> None:
        for index, match in enumerate(pattern.finditer(text)):
            output.append(
                StandardEnglishExtractor._mention(
                    key=f"{prefix}:{index}:{match.start()}",
                    start=match.start(),
                    end=match.end(),
                    entity_type=entity_type,
                    score=score,
                    text=text,
                )
            )

    def _email_claims(
        self,
        document: Document,
        existing: list[MentionCandidate],
        temporals: list[TemporalCandidate],
    ) -> tuple[list[MentionCandidate], tuple[ClaimCandidate, ...]]:
        headers_obj = document.metadata.get("headers")
        if not isinstance(headers_obj, Mapping):
            return [], ()
        headers = cast(Mapping[object, object], headers_obj)
        address_mentions: dict[str, list[MentionCandidate]] = {}
        added: list[MentionCandidate] = []
        for field in ("from", "to", "cc", "bcc"):
            values_obj = headers.get(field, ())
            values = self._string_values(values_obj)
            if not values:
                continue
            segment_ranges = self._normalized_field_ranges(document, field)
            field_mentions: list[MentionCandidate] = []
            for display, address in getaddresses(values):
                if not address:
                    continue
                found = self._find_in_ranges(document.normalized_text, address, segment_ranges)
                if found is None:
                    continue
                start, end = found
                email_mention = self._mention(
                    key=f"header:{field}:email:{start}",
                    start=start,
                    end=end,
                    entity_type=EntityType.OTHER,
                    score=1.0,
                    text=document.normalized_text,
                    normalized=f"email:{address.casefold()}",
                )
                added.append(email_mention)
                field_mentions.append(email_mention)
                if display:
                    display_found = self._find_in_ranges(
                        document.normalized_text, display, segment_ranges
                    )
                    if display_found is not None:
                        name_start, name_end = display_found
                        added.append(
                            self._mention(
                                key=f"header:{field}:name:{name_start}",
                                start=name_start,
                                end=name_end,
                                entity_type=EntityType.PERSON,
                                score=0.96,
                                text=document.normalized_text,
                            )
                        )
            address_mentions[field] = field_mentions

        all_mentions = self._dedupe_mentions([*existing, *added])
        # Restore field references after deduplication by exact span/type.
        by_signature = {
            (item.normalized_start, item.normalized_end, item.entity_type): item
            for item in all_mentions
        }
        for field, items in address_mentions.items():
            address_mentions[field] = [
                by_signature[(item.normalized_start, item.normalized_end, item.entity_type)]
                for item in items
            ]

        senders = address_mentions.get("from", [])
        recipients = [
            *address_mentions.get("to", []),
            *address_mentions.get("cc", []),
            *address_mentions.get("bcc", []),
        ]
        if not senders or not recipients:
            return added, ()
        date_indexes = self._temporal_indexes_for_fields(document, temporals, {"date"})
        claims: list[ClaimCandidate] = []
        for sender in senders:
            for recipient in recipients:
                start = min(sender.normalized_start, recipient.normalized_start)
                end = max(sender.normalized_end, recipient.normalized_end)
                participants = (
                    CandidateParticipant(role="source", mention_key=sender.key),
                    CandidateParticipant(role="target", mention_key=recipient.key),
                )
                suffix = f"{sender.normalized_start}:{recipient.normalized_start}"
                claims.append(
                    ClaimCandidate(
                        key=f"email-sent:{suffix}",
                        kind=ClaimKind.RELATIONSHIP,
                        predicate="SENT",
                        normalized_start=start,
                        normalized_end=end,
                        participants=participants,
                        polarity=Polarity.AFFIRMED,
                        modality=Modality.ASSERTED,
                        score=0.995,
                        temporal_keys=date_indexes,
                    )
                )
                claims.append(
                    ClaimCandidate(
                        key=f"email-received:{suffix}",
                        kind=ClaimKind.RELATIONSHIP,
                        predicate="RECEIVED",
                        normalized_start=start,
                        normalized_end=end,
                        participants=(
                            CandidateParticipant(role="source", mention_key=recipient.key),
                            CandidateParticipant(role="target", mention_key=sender.key),
                        ),
                        polarity=Polarity.AFFIRMED,
                        modality=Modality.ASSERTED,
                        score=0.995,
                        temporal_keys=date_indexes,
                    )
                )
                claims.append(
                    ClaimCandidate(
                        key=f"email-event:{suffix}",
                        kind=ClaimKind.EVENT,
                        predicate="EMAIL_SENT",
                        normalized_start=start,
                        normalized_end=end,
                        participants=(
                            CandidateParticipant(role="sender", mention_key=sender.key),
                            CandidateParticipant(role="recipient", mention_key=recipient.key),
                        ),
                        polarity=Polarity.AFFIRMED,
                        modality=Modality.ASSERTED,
                        score=0.995,
                        temporal_keys=date_indexes,
                    )
                )
        return added, tuple(claims)

    def _semantic_claims(
        self,
        document: Document,
        mentions: list[MentionCandidate],
        temporals: list[TemporalCandidate],
    ) -> tuple[ClaimCandidate, ...]:
        text = document.normalized_text
        claims: list[ClaimCandidate] = []
        for predicate, pattern in _RELATION_PATTERNS:
            for index, match in enumerate(pattern.finditer(text)):
                source = self._ensure_mention(
                    mentions,
                    text,
                    *match.span("source"),
                    EntityType.PERSON,
                    0.93,
                    f"rel-source:{predicate}:{index}",
                )
                target = self._ensure_mention(
                    mentions,
                    text,
                    *match.span("target"),
                    EntityType.ORGANIZATION,
                    0.94,
                    f"rel-target:{predicate}:{index}",
                )
                polarity, modality = self._qualifiers(match.group("qualifier"))
                claims.append(
                    ClaimCandidate(
                        key=f"relation:{predicate}:{index}:{match.start()}",
                        kind=ClaimKind.RELATIONSHIP,
                        predicate=predicate,
                        normalized_start=match.start(),
                        normalized_end=match.end(),
                        participants=(
                            CandidateParticipant(role="source", mention_key=source.key),
                            CandidateParticipant(role="target", mention_key=target.key),
                        ),
                        polarity=polarity,
                        modality=modality,
                        score=0.94,
                        temporal_keys=self._temporal_indexes_in_range(
                            temporals, match.start(), match.end()
                        ),
                    )
                )

        for index, match in enumerate(_EVENT_PATTERN.finditer(text)):
            if not self._explicit_subject(match.group("source")):
                # General pronoun/coreference resolution is intentionally out of
                # scope. A sentence such as "He approved it" is preserved only
                # as source text and does not become a claim.
                continue
            source_type = self._type_for_surface(match.group("source"))
            source = self._ensure_mention(
                mentions,
                text,
                *match.span("source"),
                source_type,
                0.90,
                f"event-source:{index}",
            )
            polarity, modality = self._qualifiers(match.group("qualifier"))
            predicate = self._canonical_predicate(match.group("verb"))
            target_start, target_end = match.span("target")
            target_mentions = tuple(
                sorted(
                    (
                        mention
                        for mention in mentions
                        if mention.normalized_start >= target_start
                        and mention.normalized_end <= target_end
                    ),
                    key=lambda mention: (
                        mention.normalized_start,
                        mention.normalized_end,
                        mention.entity_type.value,
                        mention.key,
                    ),
                )
            )
            object_participants = tuple(
                CandidateParticipant(role="object", mention_key=mention.key)
                for mention in target_mentions
            )
            description = match.group("target").strip()
            claims.append(
                ClaimCandidate(
                    key=f"event:{predicate}:{index}:{match.start()}",
                    kind=ClaimKind.EVENT,
                    predicate=predicate,
                    normalized_start=match.start(),
                    normalized_end=match.end(),
                    participants=(
                        CandidateParticipant(role="actor", mention_key=source.key),
                        *object_participants,
                        CandidateParticipant(role="description", literal=description),
                    ),
                    polarity=polarity,
                    modality=modality,
                    score=0.92,
                    temporal_keys=self._temporal_indexes_in_sentence(
                        text, temporals, match.start(), match.end()
                    ),
                )
            )

        for index, match in enumerate(_MEETING_PATTERN.finditer(text)):
            source = self._ensure_mention(
                mentions,
                text,
                *match.span("source"),
                EntityType.PERSON,
                0.92,
                f"meeting-source:{index}",
            )
            target = self._ensure_mention(
                mentions,
                text,
                *match.span("target"),
                EntityType.PERSON,
                0.92,
                f"meeting-target:{index}",
            )
            polarity, modality = self._qualifiers(match.group("qualifier"))
            claims.append(
                ClaimCandidate(
                    key=f"event:MET:{index}:{match.start()}",
                    kind=ClaimKind.EVENT,
                    predicate="MET",
                    normalized_start=match.start(),
                    normalized_end=match.end(),
                    participants=(
                        CandidateParticipant(role="participant", mention_key=source.key),
                        CandidateParticipant(role="participant", mention_key=target.key),
                    ),
                    polarity=polarity,
                    modality=modality,
                    score=0.94,
                    temporal_keys=self._temporal_indexes_in_sentence(
                        text, temporals, match.start(), match.end()
                    ),
                )
            )
        return tuple(claims)

    @staticmethod
    def _mention(
        *,
        key: str,
        start: int,
        end: int,
        entity_type: EntityType,
        score: float,
        text: str,
        normalized: str | None = None,
    ) -> MentionCandidate:
        surface = text[start:end]
        return MentionCandidate(
            key=key,
            normalized_start=start,
            normalized_end=end,
            entity_type=entity_type,
            score=score,
            normalized_text=normalized or " ".join(surface.casefold().split()),
        )

    @staticmethod
    def _dedupe_mentions(mentions: Iterable[MentionCandidate]) -> list[MentionCandidate]:
        selected: dict[tuple[int, int, EntityType], MentionCandidate] = {}
        for mention in mentions:
            signature = (mention.normalized_start, mention.normalized_end, mention.entity_type)
            previous = selected.get(signature)
            if previous is None or (-mention.score, mention.key) < (-previous.score, previous.key):
                selected[signature] = mention
        return sorted(
            selected.values(),
            key=lambda item: (
                item.normalized_start,
                item.normalized_end,
                item.entity_type.value,
                item.key,
            ),
        )

    @staticmethod
    def _ensure_mention(
        mentions: list[MentionCandidate],
        text: str,
        start: int,
        end: int,
        entity_type: EntityType,
        score: float,
        key: str,
    ) -> MentionCandidate:
        for mention in mentions:
            if (
                mention.normalized_start == start
                and mention.normalized_end == end
                and mention.entity_type == entity_type
            ):
                if mention.score < score:
                    upgraded = StandardEnglishExtractor._mention(
                        key=key,
                        start=start,
                        end=end,
                        entity_type=entity_type,
                        score=score,
                        text=text,
                    )
                    mentions[mentions.index(mention)] = upgraded
                    return upgraded
                return mention
        mention = StandardEnglishExtractor._mention(
            key=key,
            start=start,
            end=end,
            entity_type=entity_type,
            score=score,
            text=text,
        )
        mentions.append(mention)
        return mention

    @staticmethod
    def _qualifiers(text: str) -> tuple[Polarity, Modality]:
        lowered = text.casefold()
        polarity = (
            Polarity.NEGATED
            if re.search(r"\b(?:not|never|doesn't|didn't|hasn't|hadn't)\b", lowered)
            else Polarity.AFFIRMED
        )
        if re.search(r"\b(?:allegedly|reportedly)\b", lowered):
            modality = Modality.ALLEGED
        elif re.search(r"\b(?:may|might|could)\b", lowered):
            modality = Modality.POSSIBLE
        elif re.search(r"\bif\b", lowered):
            modality = Modality.CONDITIONAL
        else:
            modality = Modality.ASSERTED
        return polarity, modality

    @staticmethod
    def _type_for_surface(surface: str) -> EntityType:
        if re.search(
            r"\b(?:Inc\.?|LLC|Ltd\.?|Corp\.?|Corporation|Company|Bank|University)\b",
            surface,
            re.I,
        ):
            return EntityType.ORGANIZATION
        return EntityType.PERSON

    @staticmethod
    def _canonical_predicate(verb: str) -> str:
        return {
            "approve": "APPROVED",
            "approved": "APPROVED",
            "reject": "REJECTED",
            "rejected": "REJECTED",
            "authorize": "AUTHORIZED",
            "authorized": "AUTHORIZED",
            "sign": "SIGNED",
            "signed": "SIGNED",
            "execute": "EXECUTED",
            "executed": "EXECUTED",
            "file": "FILED",
            "filed": "FILED",
            "submit": "SUBMITTED",
            "submitted": "SUBMITTED",
            "pay": "PAID",
            "paid": "PAID",
            "transfer": "TRANSFERRED",
            "transferred": "TRANSFERRED",
            "purchase": "PURCHASED",
            "purchased": "PURCHASED",
            "sell": "SOLD",
            "sold": "SOLD",
            "email": "EMAILED",
            "emailed": "EMAILED",
            "notify": "NOTIFIED",
            "notified": "NOTIFIED",
        }[verb.casefold()]

    @staticmethod
    def _explicit_subject(surface: str) -> bool:
        words = surface.split()
        if surface.casefold() in {
            "he",
            "she",
            "they",
            "it",
            "we",
            "i",
            "you",
            "this",
            "that",
        }:
            return False
        if len(words) >= 2 and all(word[:1].isupper() for word in words):
            return True
        return len(words) == 1 and len(surface) >= 2 and surface.isupper()

    @staticmethod
    def _normalized_field_ranges(document: Document, field: str) -> tuple[tuple[int, int], ...]:
        mapping = document.normalized_to_raw
        ranges: list[tuple[int, int]] = []
        for segment in document.segments:
            if segment.kind != "field" or segment.field != field:
                continue
            start = bisect.bisect_left(mapping, segment.raw_start)
            end = bisect.bisect_left(mapping, segment.raw_end)
            ranges.append((start, end))
        return tuple(ranges)

    @staticmethod
    def _find_in_ranges(
        text: str,
        needle: str,
        ranges: tuple[tuple[int, int], ...],
    ) -> tuple[int, int] | None:
        for start, end in ranges:
            position = text.casefold().find(needle.casefold(), start, end)
            if position >= 0:
                return position, position + len(needle)
        return None

    @staticmethod
    def _string_values(value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, (tuple, list)):
            items = cast(tuple[object, ...] | list[object], value)
            return [item for item in items if isinstance(item, str)]
        return []

    @classmethod
    def _temporal_indexes_for_fields(
        cls,
        document: Document,
        temporals: list[TemporalCandidate],
        fields: set[str],
    ) -> tuple[int, ...]:
        ranges = [
            item
            for field in sorted(fields)
            for item in cls._normalized_field_ranges(document, field)
        ]
        return tuple(
            index
            for index, temporal in enumerate(temporals)
            if any(
                temporal.normalized_start >= start and temporal.normalized_end <= end
                for start, end in ranges
            )
        )

    @staticmethod
    def _temporal_indexes_in_range(
        temporals: list[TemporalCandidate], start: int, end: int
    ) -> tuple[int, ...]:
        return tuple(
            index
            for index, temporal in enumerate(temporals)
            if temporal.normalized_start >= start and temporal.normalized_end <= end
        )

    @classmethod
    def _temporal_indexes_in_sentence(
        cls,
        text: str,
        temporals: list[TemporalCandidate],
        start: int,
        end: int,
    ) -> tuple[int, ...]:
        sentence_start = max(text.rfind(".", 0, start), text.rfind("\n", 0, start)) + 1
        punctuation = [
            position for token in (".", "\n") if (position := text.find(token, end)) >= 0
        ]
        sentence_end = min(punctuation) if punctuation else len(text)
        return cls._temporal_indexes_in_range(temporals, sentence_start, sentence_end)
