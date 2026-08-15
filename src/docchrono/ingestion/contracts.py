from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from docchrono.domain import (
    CaseConfig,
    Document,
    DocumentBuildResult,
    DocumentFailure,
    DocumentFailureCode,
    DocumentSegment,
    SourceReference,
)


def _empty_metadata() -> Mapping[str, Any]:
    return {}


@dataclass(frozen=True, slots=True)
class LoadRequest:
    """The complete, immutable input supplied to a document loader."""

    path: Path
    content: bytes
    content_sha256: str
    suffix: str
    config: CaseConfig


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """Loader-neutral parsed content whose offsets address ``raw_text``."""

    raw_text: str
    media_type: str
    segments: tuple[DocumentSegment, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=_empty_metadata)
    warnings: tuple[str, ...] = ()


@runtime_checkable
class DocumentLoader(Protocol):
    """Public adapter contract for deterministic, side-effect-free parsing.

    Implementations must derive their result only from ``LoadRequest`` and must
    return identical semantic output for identical inputs and versions.
    """

    @property
    def name(self) -> str: ...

    @property
    def version(self) -> str: ...

    @property
    def supported_suffixes(self) -> tuple[str, ...]: ...

    @property
    def media_type(self) -> str: ...

    def load(self, request: LoadRequest) -> LoadedDocument:
        """Parse one in-memory source without executing embedded content."""

        ...


class LoaderFailure(Exception):
    """A loader's safe, typed rejection of a source."""

    def __init__(self, code: DocumentFailureCode, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class IngestionResult:
    """Complete parse-stage output, including successes and typed failures."""

    source_references: tuple[SourceReference, ...]
    documents: tuple[Document, ...]
    document_results: tuple[DocumentBuildResult, ...]
    discovery_failures: tuple[DocumentFailure, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def failures(self) -> tuple[DocumentFailure, ...]:
        per_source = tuple(
            failure for result in self.document_results for failure in result.failures
        )
        return self.discovery_failures + per_source

    @property
    def succeeded(self) -> bool:
        return bool(self.documents) and not self.failures

    @property
    def results(self) -> tuple[DocumentBuildResult, ...]:
        """Short alias useful when constructing a build report."""

        return self.document_results


__all__ = [
    "DocumentLoader",
    "IngestionResult",
    "LoadRequest",
    "LoadedDocument",
    "LoaderFailure",
]
