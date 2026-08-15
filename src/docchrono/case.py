from __future__ import annotations

from collections.abc import Iterable
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Self, cast, final

from docchrono.domain import (
    BuildReport,
    CaseConfig,
    CaseData,
    Claim,
    Document,
    Entity,
    Event,
    EvidenceSpan,
    Mention,
    Relationship,
    ReviewDecision,
    ReviewItem,
    SourceReference,
)

if TYPE_CHECKING:
    from docchrono.review import ReviewBook
    from docchrono.views import EvidenceGraph, Timeline


@final
class Case:
    """An immutable, complete snapshot of a DocChrono analysis."""

    __slots__ = ("_data",)

    def __init__(self, data: CaseData) -> None:
        self._data = data

    @classmethod
    def _from_data(cls, data: CaseData) -> Self:
        return cls(data)

    @classmethod
    def build(
        cls,
        sources: str | PathLike[str] | Iterable[str | PathLike[str]],
        *,
        config: CaseConfig | None = None,
        strict: bool = False,
    ) -> Self:
        from docchrono.advanced import Pipeline

        return cast(Self, Pipeline.standard(config=config).build(sources, strict=strict))

    @classmethod
    def load(cls, path: str | PathLike[str], *, max_bytes: int | None = None) -> Self:
        from docchrono.persistence import load_case_data

        if max_bytes is None:
            return cls(load_case_data(Path(path)))
        return cls(load_case_data(Path(path), max_bytes=max_bytes))

    @property
    def data(self) -> CaseData:
        return self._data

    @property
    def config(self) -> CaseConfig:
        return self._data.config

    @property
    def source_references(self) -> tuple[SourceReference, ...]:
        return self._data.source_references

    @property
    def documents(self) -> tuple[Document, ...]:
        return self._data.documents

    @property
    def evidence_spans(self) -> tuple[EvidenceSpan, ...]:
        return self._data.evidence_spans

    @property
    def mentions(self) -> tuple[Mention, ...]:
        return self._data.mentions

    @property
    def claims(self) -> tuple[Claim, ...]:
        return self._data.claims

    @property
    def entities(self) -> tuple[Entity, ...]:
        return self._data.entities

    @property
    def events(self) -> tuple[Event, ...]:
        return self._data.events

    @property
    def relationships(self) -> tuple[Relationship, ...]:
        return self._data.relationships

    @property
    def review_items(self) -> tuple[ReviewItem, ...]:
        return self._data.review_items

    @property
    def review_decisions(self) -> tuple[ReviewDecision, ...]:
        return self._data.review_decisions

    @property
    def report(self) -> BuildReport:
        return self._data.report

    @property
    def timeline(self) -> Timeline:
        from docchrono.views import Timeline

        return Timeline(self._data.events)

    @property
    def graph(self) -> EvidenceGraph:
        from docchrono.views import EvidenceGraph

        return EvidenceGraph(
            self._data.entities,
            self._data.events,
            self._data.relationships,
        )

    @property
    def review(self) -> ReviewBook[Case]:
        from docchrono.review import ReviewBook

        return ReviewBook(self._data, case_factory=type(self)._from_data)

    def evidence(
        self,
        item: EvidenceSpan | Mention | Claim | Entity | Event | Relationship | str,
    ) -> tuple[EvidenceSpan, ...]:
        evidence_by_id = {span.id: span for span in self._data.evidence_spans}
        if isinstance(item, EvidenceSpan):
            return (item,)
        if isinstance(item, Mention):
            ids = (item.evidence_span_id,)
        elif isinstance(item, Claim):
            ids = item.evidence_span_ids
        elif isinstance(item, Entity):
            mention_by_id = {mention.id: mention for mention in self._data.mentions}
            ids = tuple(
                mention_by_id[mention_id].evidence_span_id
                for mention_id in item.mention_ids
                if mention_id in mention_by_id
            )
        elif isinstance(item, Event):
            claim_by_id = {claim.id: claim for claim in self._data.claims}
            ids = tuple(
                evidence_id
                for claim_id in item.claim_ids
                if claim_id in claim_by_id
                for evidence_id in claim_by_id[claim_id].evidence_span_ids
            )
        elif isinstance(item, Relationship):
            claim_by_id = {claim.id: claim for claim in self._data.claims}
            ids = tuple(
                evidence_id
                for claim_id in (*item.supporting_claim_ids, *item.opposing_claim_ids)
                if claim_id in claim_by_id
                for evidence_id in claim_by_id[claim_id].evidence_span_ids
            )
        else:
            return self._evidence_for_id(item)
        return tuple(
            evidence_by_id[evidence_id]
            for evidence_id in dict.fromkeys(ids)
            if evidence_id in evidence_by_id
        )

    def _evidence_for_id(self, item_id: str) -> tuple[EvidenceSpan, ...]:
        for collection in (
            self._data.evidence_spans,
            self._data.mentions,
            self._data.claims,
            self._data.entities,
            self._data.events,
            self._data.relationships,
        ):
            for item in collection:
                if item.id == item_id:
                    return self.evidence(item)
        return ()

    def save(self, path: str | PathLike[str]) -> Path:
        from docchrono.persistence import save_case_data

        return save_case_data(self._data, Path(path))

    def save_sanitized(self, path: str | PathLike[str]) -> Path:
        from docchrono.persistence import save_case_data

        return save_case_data(self._data, Path(path), sanitized=True)
