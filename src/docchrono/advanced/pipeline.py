from __future__ import annotations

import platform
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from importlib import metadata
from os import PathLike
from typing import TYPE_CHECKING, Protocol, TypeVar

from docchrono.domain import (
    BuildManifest,
    BuildReport,
    BuildStage,
    CaseConfig,
    CaseData,
    Document,
    DocumentBuildResult,
    DocumentFailure,
    DocumentFailureCode,
    ReviewItem,
)
from docchrono.domain.ids import stable_id
from docchrono.errors import BuildFailed, ResourceLimitError, SourceError
from docchrono.extraction import ExtractionEngine, ExtractionResult, Extractor
from docchrono.ingestion import DocumentLoader, IngestionResult, Ingestor
from docchrono.resolution import EntityResolver

if TYPE_CHECKING:
    from docchrono.case import Case


_DEPENDENCIES = (
    "beautifulsoup4",
    "dateparser",
    "networkx",
    "pdfplumber",
    "pydantic",
    "python-docx",
    "rapidfuzz",
    "spacy",
)


class _HasId(Protocol):
    id: str


T = TypeVar("T", bound=_HasId)
U = TypeVar("U")


@dataclass(frozen=True, slots=True)
class Pipeline:
    """Deterministic orchestration for DocChrono's standard build stages."""

    config: CaseConfig
    _ingestor: Ingestor
    _extractor: ExtractionEngine
    _resolver: EntityResolver

    @classmethod
    def standard(cls, *, config: CaseConfig | None = None) -> Pipeline:
        return cls(
            config=config or CaseConfig(),
            _ingestor=Ingestor(),
            _extractor=ExtractionEngine.standard(),
            _resolver=EntityResolver(),
        )

    def with_loader(self, loader: DocumentLoader, *, replace_existing: bool = False) -> Pipeline:
        """Return a pipeline with one additional loader adapter."""

        from docchrono.ingestion import LoaderRegistry

        registry = LoaderRegistry(self._ingestor.registry.loaders)
        registry.register(loader, replace=replace_existing)
        return replace(self, _ingestor=Ingestor(registry))

    def with_extractor(self, extractor: Extractor, *, replace_existing: bool = False) -> Pipeline:
        """Return a pipeline with one additional extraction adapter."""

        from docchrono.extraction import ExtractorRegistry

        registry = ExtractorRegistry(self._extractor.registry.extractors)
        registry.register(extractor, replace=replace_existing)
        return replace(self, _extractor=ExtractionEngine(registry=registry))

    def build(
        self,
        sources: str | PathLike[str] | Iterable[str | PathLike[str]],
        *,
        strict: bool = False,
    ) -> Case:
        """Run every stage and return a complete immutable case snapshot."""

        from docchrono.case import Case

        ingestion = self._ingestor.ingest(sources, config=self.config)
        manifest = self._manifest(self._ingestor.adapter_versions)
        parse_report = self._report(
            ingestion,
            manifest=manifest,
            completed_stage=BuildStage.PARSE if ingestion.documents else None,
        )
        if not ingestion.documents:
            if not ingestion.source_references and ingestion.discovery_failures:
                details = "; ".join(failure.message for failure in ingestion.discovery_failures)
                raise SourceError(details)
            raise BuildFailed("no supported document could be parsed", parse_report)
        if strict and ingestion.failures:
            raise BuildFailed("document ingestion failed in strict mode", parse_report)

        extraction, document_results = self._extract_documents(ingestion, strict=strict)
        adapter_versions = dict(self._ingestor.adapter_versions)
        adapter_versions.update(dict(extraction.adapter_versions))
        manifest = self._manifest(adapter_versions)

        try:
            resolution = self._resolver.resolve(extraction, config=self.config)
        except Exception as exc:
            failure = DocumentFailure(
                code=(
                    DocumentFailureCode.LIMIT_EXCEEDED
                    if isinstance(exc, ResourceLimitError)
                    else DocumentFailureCode.EXTRACTION_FAILED
                ),
                message=f"entity resolution failed: {exc}",
                stage=BuildStage.RESOLVE,
                exception_type=type(exc).__name__,
            )
            report = self._report(
                ingestion,
                manifest=manifest,
                completed_stage=BuildStage.EXTRACT,
                document_results=document_results,
                global_failures=(*ingestion.discovery_failures, failure),
                review_items=extraction.review_items,
            )
            raise BuildFailed("entity resolution failed", report) from exc

        review_items = _unique_by_id((*extraction.review_items, *resolution.review_items))
        completed_results = tuple(
            result.model_copy(
                update={
                    "completed_stage": (
                        BuildStage.GRAPH if result.succeeded else result.completed_stage
                    )
                }
            )
            for result in document_results
        )
        report = self._report(
            ingestion,
            manifest=manifest,
            completed_stage=BuildStage.GRAPH,
            document_results=completed_results,
            review_items=review_items,
        )
        data = CaseData(
            config=self.config,
            source_references=ingestion.source_references,
            documents=ingestion.documents,
            evidence_spans=extraction.evidence_spans,
            mentions=extraction.mentions,
            claims=resolution.claims,
            entities=resolution.entities,
            events=resolution.events,
            relationships=resolution.relationships,
            review_items=review_items,
            report=report,
        )
        return Case(data)

    def _extract_documents(
        self,
        ingestion: IngestionResult,
        *,
        strict: bool,
    ) -> tuple[ExtractionResult, tuple[DocumentBuildResult, ...]]:
        results: list[ExtractionResult] = []
        document_results = list(ingestion.document_results)
        for document in sorted(ingestion.documents, key=lambda item: item.id):
            try:
                results.append(self._extractor.extract((document,), config=self.config))
            except Exception as exc:
                failure = DocumentFailure(
                    code=DocumentFailureCode.EXTRACTION_FAILED,
                    message=str(exc) or "document extraction failed",
                    source_reference_id=(
                        document.source_reference_ids[0] if document.source_reference_ids else None
                    ),
                    stage=BuildStage.EXTRACT,
                    exception_type=type(exc).__name__,
                )
                document_results = _attach_failure(document_results, document, failure)
                if strict:
                    partial = _combine_extractions(results)
                    report = self._report(
                        ingestion,
                        manifest=self._manifest(self._ingestor.adapter_versions),
                        completed_stage=BuildStage.EXTRACT,
                        document_results=tuple(document_results),
                        review_items=partial.review_items,
                    )
                    raise BuildFailed("document extraction failed in strict mode", report) from exc
        return _combine_extractions(results), tuple(document_results)

    def _manifest(self, adapter_versions: Mapping[str, str]) -> BuildManifest:
        return BuildManifest(
            docchrono_version=_installed_version("docchrono", fallback="0.1.0rc1"),
            python_version=platform.python_version(),
            platform=sys.platform,
            dependency_versions={name: _installed_version(name) for name in _DEPENDENCIES},
            adapter_versions=dict(sorted(adapter_versions.items())),
            model_versions=self._extractor.model_versions,
            config_fingerprint=stable_id("config", self.config),
        )

    @staticmethod
    def _report(
        ingestion: IngestionResult,
        *,
        manifest: BuildManifest,
        completed_stage: BuildStage | None,
        document_results: tuple[DocumentBuildResult, ...] | None = None,
        global_failures: tuple[DocumentFailure, ...] | None = None,
        review_items: tuple[ReviewItem, ...] = (),
    ) -> BuildReport:
        return BuildReport(
            requested_stage=BuildStage.GRAPH,
            completed_stage=completed_stage,
            documents=document_results or ingestion.document_results,
            failures=(ingestion.discovery_failures if global_failures is None else global_failures),
            warnings=ingestion.warnings,
            review_items=review_items,
            manifest=manifest,
        )


def _installed_version(distribution: str, *, fallback: str = "unknown") -> str:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return fallback


def _combine_extractions(results: Iterable[ExtractionResult]) -> ExtractionResult:
    values = tuple(results)
    return ExtractionResult(
        evidence_spans=_unique_by_id(item for result in values for item in result.evidence_spans),
        mentions=_unique_by_id(item for result in values for item in result.mentions),
        temporal_expressions=_unique_temporals(
            item for result in values for item in result.temporal_expressions
        ),
        claims=_unique_by_id(item for result in values for item in result.claims),
        review_items=_unique_by_id(item for result in values for item in result.review_items),
        accepted_mention_ids=_unique_strings(
            item for result in values for item in result.accepted_mention_ids
        ),
        review_mention_ids=_unique_strings(
            item for result in values for item in result.review_mention_ids
        ),
        deferred_mention_ids=_unique_strings(
            item for result in values for item in result.deferred_mention_ids
        ),
        accepted_claim_ids=_unique_strings(
            item for result in values for item in result.accepted_claim_ids
        ),
        review_claim_ids=_unique_strings(
            item for result in values for item in result.review_claim_ids
        ),
        deferred_claim_ids=_unique_strings(
            item for result in values for item in result.deferred_claim_ids
        ),
        adapter_versions=tuple(
            sorted({item for result in values for item in result.adapter_versions})
        ),
    )


def _unique_by_id(items: Iterable[T]) -> tuple[T, ...]:
    by_id = {item.id: item for item in items}
    return tuple(by_id[key] for key in sorted(by_id))


def _unique_temporals(items: Iterable[U]) -> tuple[U, ...]:
    by_id = {stable_id("temporal", item): item for item in items}
    return tuple(by_id[key] for key in sorted(by_id))


def _unique_strings(items: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(set(items)))


def _attach_failure(
    results: list[DocumentBuildResult],
    document: Document,
    failure: DocumentFailure,
) -> list[DocumentBuildResult]:
    updated: list[DocumentBuildResult] = []
    matched = False
    for result in results:
        if result.document_id == document.id:
            matched = True
            updated.append(
                result.model_copy(
                    update={
                        "failures": (*result.failures, failure),
                        "completed_stage": BuildStage.PARSE,
                    }
                )
            )
        else:
            updated.append(result)
    if not matched:
        updated.append(
            DocumentBuildResult(
                source_reference_id=failure.source_reference_id or "unknown",
                document_id=document.id,
                completed_stage=BuildStage.PARSE,
                failures=(failure,),
            )
        )
    return updated
