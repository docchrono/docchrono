from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from docchrono.domain import (
    BuildStage,
    CaseConfig,
    Document,
    DocumentBuildResult,
    DocumentFailure,
    DocumentFailureCode,
    DocumentSegment,
    SourceReference,
)
from docchrono.domain.ids import stable_id
from docchrono.errors import AdapterContractError

from .contracts import IngestionResult, LoadedDocument, LoaderFailure, LoadRequest
from .normalization import normalize_text
from .registry import LoaderRegistry

PathInput = str | os.PathLike[str]


@dataclass(frozen=True, slots=True)
class _SourceCandidate:
    path: Path
    reference: SourceReference
    suffix: str


@dataclass(frozen=True, slots=True)
class _DiscoveredSource:
    path: Path
    root: Path
    attributes: os.stat_result


class _SourceReadFailure(Exception):
    def __init__(self, code: DocumentFailureCode, message: str) -> None:
        super().__init__(message)
        self.code = code


class Ingestor:
    """Discover, fingerprint, deduplicate, and parse local documents."""

    def __init__(self, registry: LoaderRegistry | None = None) -> None:
        self.registry = registry if registry is not None else LoaderRegistry.with_builtins()

    @property
    def adapter_versions(self) -> dict[str, str]:
        return self.registry.adapter_versions

    def ingest(
        self,
        paths: PathInput | Iterable[PathInput],
        config: CaseConfig | None = None,
    ) -> IngestionResult:
        effective_config = config if config is not None else CaseConfig()
        requested_paths = _coerce_paths(paths)
        discovered, discovery_failures, discovery_warnings = _discover_paths(
            requested_paths,
            config=effective_config,
        )

        source_references: list[SourceReference] = []
        candidates_by_digest: dict[str, list[_SourceCandidate]] = {}
        content_by_digest: dict[str, bytes] = {}
        failures = list(discovery_failures)
        global_warnings = list(discovery_warnings)

        for discovered_source in discovered:
            path = discovered_source.path
            try:
                digest, size_bytes, content = _read_and_fingerprint(
                    discovered_source,
                    limit=effective_config.max_file_bytes,
                    follow_symlinks=effective_config.follow_symlinks,
                )
            except _SourceReadFailure as exc:
                failures.append(_failure(exc.code, str(exc), exception=exc))
                continue
            except OSError as exc:
                failures.append(
                    _failure(
                        DocumentFailureCode.SOURCE_UNAVAILABLE,
                        f"Source could not be read: {path}: {exc}",
                        exception=exc,
                    )
                )
                continue

            suffix = path.suffix.lower()
            loader = self.registry.loader_for(suffix)
            media_type = loader.media_type if loader is not None else "application/octet-stream"
            reference = SourceReference(
                id=stable_id("src", path.as_posix(), digest),
                path=path.as_posix(),
                filename=path.name,
                media_type=media_type,
                size_bytes=size_bytes,
                content_sha256=digest,
                metadata={"suffix": suffix},
            )
            source_references.append(reference)
            candidate = _SourceCandidate(
                path=path,
                reference=reference,
                suffix=suffix,
            )
            candidates_by_digest.setdefault(digest, []).append(candidate)
            if digest not in content_by_digest:
                content_by_digest[digest] = content

        documents: list[Document] = []
        results_by_source: dict[str, DocumentBuildResult] = {}

        for digest, group in candidates_by_digest.items():
            reference_ids = tuple(candidate.reference.id for candidate in group)
            supported = [
                candidate
                for candidate in group
                if self.registry.loader_for(candidate.suffix) is not None
            ]
            if not supported:
                for candidate in group:
                    failure = _failure(
                        DocumentFailureCode.UNSUPPORTED_FORMAT,
                        (
                            f"Unsupported file suffix {candidate.suffix or '<none>'!r}: "
                            f"{candidate.path}"
                        ),
                        source_reference_id=candidate.reference.id,
                    )
                    results_by_source[candidate.reference.id] = DocumentBuildResult(
                        source_reference_id=candidate.reference.id,
                        failures=(failure,),
                    )
                continue

            selected = supported[0]
            loader = self.registry.loader_for(selected.suffix)
            assert loader is not None  # narrowed by the supported filter
            content = content_by_digest[digest]
            group_warnings: list[str] = []
            loader_names = {
                candidate_loader.name
                for candidate in supported
                if (candidate_loader := self.registry.loader_for(candidate.suffix)) is not None
            }
            if len(loader_names) > 1:
                warning = (
                    f"Byte-identical sources have conflicting formats; parsed with {loader.name}: "
                    + ", ".join(candidate.path.as_posix() for candidate in group)
                )
                group_warnings.append(warning)
                global_warnings.append(warning)
            if len(supported) != len(group):
                warning = (
                    "An unsupported-suffix source was linked to a byte-identical "
                    "supported document: "
                    + ", ".join(
                        candidate.path.as_posix()
                        for candidate in group
                        if candidate not in supported
                    )
                )
                group_warnings.append(warning)
                global_warnings.append(warning)

            try:
                loaded = loader.load(
                    LoadRequest(
                        path=selected.path,
                        content=content,
                        content_sha256=digest,
                        suffix=selected.suffix,
                        config=effective_config,
                    )
                )
                _validate_loaded_document(loaded, loader.media_type)
                if len(loaded.raw_text) > effective_config.max_extracted_chars:
                    raise LoaderFailure(
                        DocumentFailureCode.LIMIT_EXCEEDED,
                        (
                            f"Extracted text is {len(loaded.raw_text)} characters; limit is "
                            f"{effective_config.max_extracted_chars}"
                        ),
                    )
                normalized_text, offset_map = normalize_text(loaded.raw_text)
                document = Document(
                    id=stable_id("doc", digest),
                    content_sha256=digest,
                    source_reference_ids=reference_ids,
                    media_type=loaded.media_type,
                    raw_text=loaded.raw_text,
                    normalized_text=normalized_text,
                    normalized_to_raw=offset_map,
                    segments=tuple(_canonical_segments(loaded.segments)),
                    metadata=dict(loaded.metadata),
                    parser_name=loader.name,
                    parser_version=loader.version,
                )
            except LoaderFailure as exc:
                for candidate in group:
                    failure = _failure(
                        exc.code,
                        str(exc),
                        source_reference_id=candidate.reference.id,
                        exception=exc,
                    )
                    results_by_source[candidate.reference.id] = DocumentBuildResult(
                        source_reference_id=candidate.reference.id,
                        failures=(failure,),
                        warnings=tuple(group_warnings),
                    )
                continue
            except AdapterContractError as exc:
                for candidate in group:
                    failure = _failure(
                        DocumentFailureCode.ADAPTER_CONTRACT,
                        str(exc),
                        source_reference_id=candidate.reference.id,
                        exception=exc,
                    )
                    results_by_source[candidate.reference.id] = DocumentBuildResult(
                        source_reference_id=candidate.reference.id,
                        failures=(failure,),
                        warnings=tuple(group_warnings),
                    )
                continue
            except Exception as exc:
                for candidate in group:
                    failure = _failure(
                        DocumentFailureCode.LOADER_FAILED,
                        f"Loader {loader.name!r} failed: {type(exc).__name__}",
                        source_reference_id=candidate.reference.id,
                        exception=exc,
                    )
                    results_by_source[candidate.reference.id] = DocumentBuildResult(
                        source_reference_id=candidate.reference.id,
                        failures=(failure,),
                        warnings=tuple(group_warnings),
                    )
                continue

            documents.append(document)
            result_warnings = tuple(group_warnings) + loaded.warnings
            for warning in loaded.warnings:
                global_warnings.append(f"{selected.path}: {warning}")
            for candidate in group:
                results_by_source[candidate.reference.id] = DocumentBuildResult(
                    source_reference_id=candidate.reference.id,
                    document_id=document.id,
                    completed_stage=BuildStage.PARSE,
                    warnings=result_warnings,
                )

        ordered_references = tuple(
            sorted(source_references, key=lambda item: _string_path_key(item.path))
        )
        ordered_documents = tuple(sorted(documents, key=lambda item: item.content_sha256))
        ordered_results = tuple(
            results_by_source[reference.id]
            for reference in ordered_references
            if reference.id in results_by_source
        )
        return IngestionResult(
            source_references=ordered_references,
            documents=ordered_documents,
            document_results=ordered_results,
            discovery_failures=tuple(failures),
            warnings=tuple(dict.fromkeys(global_warnings)),
        )


def _coerce_paths(paths: PathInput | Iterable[PathInput]) -> tuple[Path, ...]:
    values = (paths,) if isinstance(paths, (str, os.PathLike)) else tuple(paths)
    return tuple(Path(os.path.abspath(os.fspath(value))) for value in values)


def _discover_paths(
    requested: tuple[Path, ...],
    *,
    config: CaseConfig,
) -> tuple[tuple[_DiscoveredSource, ...], tuple[DocumentFailure, ...], tuple[str, ...]]:
    files: dict[str, _DiscoveredSource] = {}
    failures: list[DocumentFailure] = []
    warnings: list[str] = []
    visited_directories: set[str] = set()
    total_bytes = 0
    traversal_exhausted = False

    if not requested:
        failures.append(
            _failure(DocumentFailureCode.SOURCE_UNAVAILABLE, "No input paths were provided")
        )
        return (), tuple(failures), ()

    def limit_failure(message: str) -> None:
        failures.append(_failure(DocumentFailureCode.LIMIT_EXCEEDED, message))

    def visit(path: Path, *, root: Path, depth: int) -> None:
        nonlocal total_bytes, traversal_exhausted
        if traversal_exhausted:
            return
        if depth > config.max_depth:
            limit_failure(f"Traversal depth exceeds limit {config.max_depth}: {path}")
            return
        try:
            attributes = path.lstat()
        except OSError as exc:
            failures.append(
                _failure(
                    DocumentFailureCode.SOURCE_UNAVAILABLE,
                    f"Source is unavailable: {path}: {exc}",
                    exception=exc,
                )
            )
            return

        is_link_like = _is_link_like(attributes)
        if is_link_like and not config.follow_symlinks:
            warnings.append(f"Skipped symlink or reparse point: {path}")
            return
        if is_link_like:
            try:
                attributes = path.stat()
            except OSError as exc:
                failures.append(
                    _failure(
                        DocumentFailureCode.SOURCE_UNAVAILABLE,
                        f"Symlink target is unavailable: {path}: {exc}",
                        exception=exc,
                    )
                )
                return

        try:
            resolved = path.resolve(strict=True)
        except OSError as exc:
            failures.append(
                _failure(
                    DocumentFailureCode.SOURCE_UNAVAILABLE,
                    f"Source could not be resolved: {path}: {exc}",
                    exception=exc,
                )
            )
            return
        if not _is_within_root(resolved, root):
            failures.append(
                _failure(
                    DocumentFailureCode.SOURCE_UNAVAILABLE,
                    f"Source resolves outside requested root: {path}",
                )
            )
            return

        if stat.S_ISDIR(attributes.st_mode):
            identity = _directory_identity(resolved, attributes)
            if identity in visited_directories:
                warnings.append(f"Skipped already visited directory: {path}")
                return
            visited_directories.add(identity)
            try:
                children = sorted(path.iterdir(), key=_path_key)
            except OSError as exc:
                failures.append(
                    _failure(
                        DocumentFailureCode.SOURCE_UNAVAILABLE,
                        f"Directory could not be listed: {path}: {exc}",
                        exception=exc,
                    )
                )
                return
            for child in children:
                visit(child, root=root, depth=depth + 1)
            return

        if not stat.S_ISREG(attributes.st_mode):
            failures.append(
                _failure(
                    DocumentFailureCode.SOURCE_UNAVAILABLE,
                    f"Source is not a regular file: {path}",
                )
            )
            return

        key = os.path.normcase(str(path))
        if key in files:
            return
        if len(files) >= config.max_files:
            limit_failure(f"Discovered file count exceeds limit {config.max_files}: {path}")
            traversal_exhausted = True
            return
        candidate_size = attributes.st_size
        if candidate_size < 0:
            failures.append(
                _failure(
                    DocumentFailureCode.SOURCE_UNAVAILABLE,
                    f"Source reported an invalid negative size: {path}",
                )
            )
            return
        if total_bytes + candidate_size > config.max_total_bytes:
            limit_failure(
                f"Discovered source bytes exceed collection limit {config.max_total_bytes}: {path}"
            )
            traversal_exhausted = True
            return
        files[key] = _DiscoveredSource(path=path, root=root, attributes=attributes)
        total_bytes += candidate_size

    for requested_path in sorted(dict.fromkeys(requested), key=_path_key):
        root = _requested_root(requested_path)
        visit(requested_path, root=root, depth=0)
    ordered = tuple(sorted(files.values(), key=lambda item: _path_key(item.path)))
    return ordered, tuple(failures), tuple(warnings)


def _read_and_fingerprint(
    discovered: _DiscoveredSource,
    *,
    limit: int,
    follow_symlinks: bool,
) -> tuple[str, int, bytes]:
    path = discovered.path
    try:
        current_attributes = path.lstat()
    except OSError as exc:
        raise _SourceReadFailure(
            DocumentFailureCode.SOURCE_UNAVAILABLE,
            f"Source became unavailable before opening: {path}: {exc}",
        ) from exc

    current_is_link = _is_link_like(current_attributes)
    if current_is_link and not follow_symlinks:
        raise _SourceReadFailure(
            DocumentFailureCode.SOURCE_UNAVAILABLE,
            f"Source became a symlink or reparse point before opening: {path}",
        )
    if current_is_link:
        try:
            current_attributes = path.stat()
        except OSError as exc:
            raise _SourceReadFailure(
                DocumentFailureCode.SOURCE_UNAVAILABLE,
                f"Source target became unavailable before opening: {path}: {exc}",
            ) from exc
    if not _same_file_snapshot(current_attributes, discovered.attributes):
        raise _SourceReadFailure(
            DocumentFailureCode.SOURCE_UNAVAILABLE,
            f"Source changed between discovery and opening: {path}",
        )
    if current_attributes.st_size > limit:
        raise _SourceReadFailure(
            DocumentFailureCode.LIMIT_EXCEEDED,
            f"Source is {current_attributes.st_size} bytes; limit is {limit}: {path}",
        )
    _require_root_containment(path, discovered.root)

    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOINHERIT", 0)
    if not follow_symlinks:
        flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise _SourceReadFailure(
            DocumentFailureCode.SOURCE_UNAVAILABLE,
            f"Source could not be opened safely: {path}: {exc}",
        ) from exc

    hasher = hashlib.sha256()
    retained: list[bytes] = []
    size = 0
    try:
        with os.fdopen(descriptor, "rb", closefd=True) as source:
            opened_attributes = os.fstat(source.fileno())
            if not stat.S_ISREG(opened_attributes.st_mode):
                raise _SourceReadFailure(
                    DocumentFailureCode.SOURCE_UNAVAILABLE,
                    f"Opened source is not a regular file: {path}",
                )
            if not _same_file_snapshot(opened_attributes, current_attributes):
                raise _SourceReadFailure(
                    DocumentFailureCode.SOURCE_UNAVAILABLE,
                    f"Source changed while it was being opened: {path}",
                )
            if opened_attributes.st_size > limit:
                raise _SourceReadFailure(
                    DocumentFailureCode.LIMIT_EXCEEDED,
                    f"Source is {opened_attributes.st_size} bytes; limit is {limit}: {path}",
                )

            while chunk := source.read(min(1024 * 1024, limit - size + 1)):
                size += len(chunk)
                if size > limit:
                    raise _SourceReadFailure(
                        DocumentFailureCode.LIMIT_EXCEEDED,
                        f"Source grew beyond byte limit {limit} while reading: {path}",
                    )
                hasher.update(chunk)
                retained.append(chunk)

            final_attributes = os.fstat(source.fileno())
            if not _stable_during_read(opened_attributes, final_attributes, size):
                raise _SourceReadFailure(
                    DocumentFailureCode.SOURCE_UNAVAILABLE,
                    f"Source changed while it was being read: {path}",
                )
    except Exception:
        # ``fdopen`` owns and closes the descriptor after it succeeds. If it
        # failed before taking ownership, close the raw descriptor here.
        with suppress(OSError):
            os.close(descriptor)
        raise

    try:
        path_attributes = path.stat() if follow_symlinks else path.lstat()
    except OSError as exc:
        raise _SourceReadFailure(
            DocumentFailureCode.SOURCE_UNAVAILABLE,
            f"Source path changed after reading: {path}: {exc}",
        ) from exc
    if not _same_file_snapshot(path_attributes, final_attributes):
        raise _SourceReadFailure(
            DocumentFailureCode.SOURCE_UNAVAILABLE,
            f"Source path was replaced while reading: {path}",
        )
    if not follow_symlinks and _is_link_like(path_attributes):
        raise _SourceReadFailure(
            DocumentFailureCode.SOURCE_UNAVAILABLE,
            f"Source path became a symlink or reparse point while reading: {path}",
        )
    _require_root_containment(path, discovered.root)
    return hasher.hexdigest(), size, b"".join(retained)


def _requested_root(path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        attributes = path.stat()
    except OSError:
        return path.parent.resolve(strict=False)
    return resolved if stat.S_ISDIR(attributes.st_mode) else resolved.parent


def _require_root_containment(path: Path, root: Path) -> None:
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise _SourceReadFailure(
            DocumentFailureCode.SOURCE_UNAVAILABLE,
            f"Source could not be resolved safely: {path}: {exc}",
        ) from exc
    if not _is_within_root(resolved, root):
        raise _SourceReadFailure(
            DocumentFailureCode.SOURCE_UNAVAILABLE,
            f"Source resolves outside requested root: {path}",
        )


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_link_like(attributes: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    file_attributes = int(getattr(attributes, "st_file_attributes", 0))
    return stat.S_ISLNK(attributes.st_mode) or bool(file_attributes & reparse_flag)


def _same_file_identity(left: os.stat_result, right: os.stat_result) -> bool:
    left_identity = (left.st_dev, left.st_ino)
    right_identity = (right.st_dev, right.st_ino)
    if left_identity != (0, 0) and right_identity != (0, 0):
        return left_identity == right_identity
    return stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode) and left.st_size == right.st_size


def _same_file_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _same_file_identity(left, right)
        and stat.S_IFMT(left.st_mode) == stat.S_IFMT(right.st_mode)
        and left.st_size == right.st_size
    )


def _stable_during_read(
    before: os.stat_result,
    after: os.stat_result,
    bytes_read: int,
) -> bool:
    return _same_file_snapshot(before, after) and before.st_size == after.st_size == bytes_read


def _directory_identity(path: Path, attributes: os.stat_result) -> str:
    if (attributes.st_dev, attributes.st_ino) != (0, 0):
        return f"{attributes.st_dev}:{attributes.st_ino}"
    return os.path.normcase(str(path))


def _validate_loaded_document(loaded: object, expected_media_type: str) -> None:
    if not isinstance(loaded, LoadedDocument):
        raise AdapterContractError("loader.load() must return LoadedDocument")
    if loaded.media_type != expected_media_type:
        raise AdapterContractError(
            f"loader returned media type {loaded.media_type!r}; expected {expected_media_type!r}"
        )
    raw_text = cast(object, loaded.raw_text)
    segments = cast(object, loaded.segments)
    metadata = cast(object, loaded.metadata)
    warnings = cast(object, loaded.warnings)
    if not isinstance(raw_text, str):
        raise AdapterContractError("LoadedDocument.raw_text must be str")
    if not isinstance(segments, tuple):
        raise AdapterContractError("LoadedDocument.segments must be a tuple of DocumentSegment")
    segment_values = cast(tuple[object, ...], segments)
    if not all(isinstance(segment, DocumentSegment) for segment in segment_values):
        raise AdapterContractError("LoadedDocument.segments must be a tuple of DocumentSegment")
    typed_segments = cast(tuple[DocumentSegment, ...], segment_values)
    for segment in typed_segments:
        if segment.raw_end > len(loaded.raw_text):
            raise AdapterContractError("loader returned a segment beyond raw_text")
    if not isinstance(metadata, Mapping):
        raise AdapterContractError("LoadedDocument.metadata must be a mapping")
    try:
        json.dumps(metadata, ensure_ascii=False, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise AdapterContractError(
            "LoadedDocument.metadata must be deterministic JSON data"
        ) from exc
    if not isinstance(warnings, tuple):
        raise AdapterContractError("LoadedDocument.warnings must be a tuple of strings")
    warning_values = cast(tuple[object, ...], warnings)
    if not all(isinstance(warning, str) for warning in warning_values):
        raise AdapterContractError("LoadedDocument.warnings must be a tuple of strings")


def _canonical_segments(segments: tuple[DocumentSegment, ...]) -> list[DocumentSegment]:
    return sorted(
        segments,
        key=lambda segment: (
            segment.raw_start,
            -segment.raw_end,
            segment.kind,
            segment.field or "",
            segment.label or "",
        ),
    )


def _failure(
    code: DocumentFailureCode,
    message: str,
    *,
    source_reference_id: str | None = None,
    exception: Exception | None = None,
) -> DocumentFailure:
    return DocumentFailure(
        code=code,
        message=message,
        source_reference_id=source_reference_id,
        stage=BuildStage.PARSE,
        exception_type=type(exception).__name__ if exception is not None else None,
    )


def _path_key(path: Path) -> tuple[str, str]:
    value = path.as_posix()
    return value.casefold(), value


def _string_path_key(path: str) -> tuple[str, str]:
    return path.casefold(), path


__all__ = ["Ingestor", "PathInput"]
