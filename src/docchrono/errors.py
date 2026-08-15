from __future__ import annotations

from docchrono.domain import BuildReport


class DocChronoError(Exception):
    """Base exception for public DocChrono failures."""


class SourceError(DocChronoError):
    pass


class ConfigurationError(DocChronoError):
    pass


class ResourceLimitError(DocChronoError):
    """Raised when a deterministic safety budget would be exceeded."""


class BuildFailed(DocChronoError):
    def __init__(self, message: str, report: BuildReport) -> None:
        super().__init__(message)
        self.report = report


class AmbiguousReferenceError(DocChronoError):
    pass


class ReviewDecisionError(DocChronoError):
    pass


class CompatibilityError(DocChronoError):
    pass


class IntegrityError(DocChronoError):
    pass


class PersistenceError(DocChronoError):
    pass


class AdapterConflictError(DocChronoError):
    pass


class AdapterContractError(DocChronoError):
    pass
