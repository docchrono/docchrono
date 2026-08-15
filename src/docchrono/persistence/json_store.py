from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
import warnings
from collections.abc import Collection, Mapping
from contextlib import suppress
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

from pydantic import ValidationError

from docchrono.domain import CaseData, Document, OffsetMap, TemporalExpression
from docchrono.errors import (
    CompatibilityError,
    IntegrityError,
    PersistenceError,
    ReviewDecisionError,
)
from docchrono.review import ReviewEngine

FORMAT_NAME = "docchrono.case"
FORMAT_VERSION = 1
SCHEMA_VERSION = "1.0"
DEFAULT_MAX_SERIALIZED_BYTES = 256 * 1024 * 1024
MAX_JSON_NESTING_DEPTH = 100
MAX_CASE_RECORDS = 250_000
SANITIZED_WARNING = (
    "Sanitized case omits full raw and normalized document text; evidence-span quotes remain."
)
_SANITIZED_METADATA_KEY = "docchrono_sanitized_text"


class SanitizedCaseWarning(UserWarning):
    """Raised when writing a sanitized case that still contains evidence quotes."""


def dumps_case_data(data: CaseData, *, sanitized: bool = False) -> bytes:
    try:
        should_sanitize = sanitized or is_sanitized_case(data)
        stored = sanitize_case_data(data, emit_warning=sanitized) if should_sanitize else data
        validate_case_data(stored, sanitized=should_sanitize)
        payload = stored.model_dump(mode="json")
        _enforce_record_budget(payload)
        payload_bytes = _canonical_json(payload)
        envelope = {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "payload": payload,
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "sanitized": should_sanitize,
            "schema_version": SCHEMA_VERSION,
        }
        return _canonical_json(envelope) + b"\n"
    except RecursionError as error:
        raise IntegrityError("case data exceeds the supported nesting depth") from error
    except MemoryError as error:
        raise PersistenceError("not enough memory to serialize case data safely") from error


def loads_case_data(
    content: bytes | str,
    *,
    max_bytes: int = DEFAULT_MAX_SERIALIZED_BYTES,
) -> CaseData:
    limit = _validate_max_bytes(max_bytes)
    try:
        if isinstance(content, bytes):
            if len(content) > limit:
                raise PersistenceError(_oversized_message(limit))
            text = content.decode("utf-8")
        else:
            # Every Unicode code point requires at least one UTF-8 byte.  This
            # cheap check avoids creating another huge allocation when the
            # caller already supplied a string that is certainly over budget.
            if len(content) > limit:
                raise PersistenceError(_oversized_message(limit))
            if len(content.encode("utf-8")) > limit:
                raise PersistenceError(_oversized_message(limit))
            text = content
    except UnicodeDecodeError as error:
        raise IntegrityError("saved case is not valid UTF-8") from error
    except MemoryError as error:
        raise PersistenceError("not enough memory to decode saved case safely") from error
    return _loads_case_text(text)


def _loads_case_text(text: str) -> CaseData:
    try:
        return _loads_case_text_impl(text)
    except RecursionError as error:
        raise IntegrityError("saved case exceeds the supported nesting depth") from error
    except MemoryError as error:
        raise PersistenceError("not enough memory to load saved case safely") from error


def _loads_case_text_impl(text: str) -> CaseData:
    _enforce_json_nesting(text)
    try:
        parsed = cast(
            object,
            json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_non_finite,
            ),
        )
    except IntegrityError:
        raise
    except RecursionError as error:
        raise IntegrityError("saved case exceeds the supported JSON nesting depth") from error
    except MemoryError as error:
        raise PersistenceError("not enough memory to parse saved case safely") from error
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise IntegrityError("saved case is not valid canonical JSON") from error

    if not isinstance(parsed, dict):
        raise IntegrityError("saved case envelope must be a JSON object")
    envelope = cast("dict[str, object]", parsed)
    if envelope.get("format") != FORMAT_NAME:
        raise CompatibilityError(f"unsupported saved case format: {envelope.get('format')!r}")
    if envelope.get("format_version") != FORMAT_VERSION:
        raise CompatibilityError(
            f"unsupported saved case format version: {envelope.get('format_version')!r}"
        )
    if envelope.get("schema_version") != SCHEMA_VERSION:
        raise CompatibilityError(
            f"unsupported saved case schema version: {envelope.get('schema_version')!r}"
        )
    sanitized = envelope.get("sanitized")
    if not isinstance(sanitized, bool):
        raise IntegrityError("saved case sanitized flag must be boolean")
    payload_value = envelope.get("payload")
    if not isinstance(payload_value, dict):
        raise IntegrityError("saved case payload must be a JSON object")
    payload = cast("dict[str, object]", payload_value)
    _enforce_record_budget(payload)
    payload_hash = envelope.get("payload_sha256")
    if not isinstance(payload_hash, str) or len(payload_hash) != 64:
        raise IntegrityError("saved case payload hash is missing or malformed")
    actual_hash = hashlib.sha256(_canonical_json(payload)).hexdigest()
    if not hmac.compare_digest(actual_hash, payload_hash):
        raise IntegrityError("saved case payload hash does not match its contents")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CompatibilityError(
            f"unsupported payload schema version: {payload.get('schema_version')!r}"
        )
    try:
        data = CaseData.model_validate(payload)
    except ValidationError as error:
        raise IntegrityError("saved case payload does not satisfy the domain schema") from error
    except RecursionError as error:
        raise IntegrityError("saved case payload exceeds the supported nesting depth") from error
    except MemoryError as error:
        raise PersistenceError("not enough memory to validate saved case safely") from error
    if sanitized != is_sanitized_case(data):
        raise IntegrityError("saved case sanitization marker does not match its payload")
    validate_case_data(data, sanitized=sanitized)
    return data


def save_case_data(
    data: CaseData,
    path: str | os.PathLike[str],
    *,
    sanitized: bool = False,
) -> Path:
    target = Path(path)
    parent = target.parent
    if not parent.exists():
        raise PersistenceError(f"save directory does not exist: {parent}")
    content = dumps_case_data(data, sanitized=sanitized)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, target)
        temporary_path = None
    except OSError as error:
        raise PersistenceError(f"could not atomically save case to {target}") from error
    finally:
        if temporary_path is not None:
            with suppress(OSError):
                temporary_path.unlink(missing_ok=True)
    return target


def load_case_data(
    path: str | os.PathLike[str],
    *,
    max_bytes: int = DEFAULT_MAX_SERIALIZED_BYTES,
) -> CaseData:
    source = Path(path)
    limit = _validate_max_bytes(max_bytes)
    try:
        with source.open("rb") as stream:
            size = os.fstat(stream.fileno()).st_size
            if size > limit:
                raise PersistenceError(_oversized_message(limit, source=source))
            content = bytearray()
            while chunk := stream.read(min(1024 * 1024, limit - len(content) + 1)):
                content.extend(chunk)
                if len(content) > limit:
                    raise PersistenceError(_oversized_message(limit, source=source))
            try:
                text = content.decode("utf-8")
            except UnicodeDecodeError as error:
                raise IntegrityError("saved case is not valid UTF-8") from error
    except (IntegrityError, PersistenceError):
        raise
    except MemoryError as error:
        raise PersistenceError(f"not enough memory to read saved case from {source}") from error
    except OSError as error:
        raise PersistenceError(f"could not read saved case from {source}") from error
    return _loads_case_text(text)


def sanitize_case_data(data: CaseData, *, emit_warning: bool = True) -> CaseData:
    if emit_warning:
        warnings.warn(SANITIZED_WARNING, SanitizedCaseWarning, stacklevel=2)
    documents: list[Document] = []
    for document in data.documents:
        metadata = dict(document.metadata)
        metadata[_SANITIZED_METADATA_KEY] = True
        documents.append(
            document.model_copy(
                update={
                    "raw_text": "",
                    "normalized_text": "",
                    "normalized_to_raw": OffsetMap.identity(0),
                    "metadata": metadata,
                }
            )
        )
    report_warnings = data.report.warnings
    if SANITIZED_WARNING not in report_warnings:
        report_warnings = (*report_warnings, SANITIZED_WARNING)
    report = data.report.model_copy(update={"warnings": report_warnings})
    return data.model_copy(update={"documents": tuple(documents), "report": report})


def is_sanitized_case(data: CaseData) -> bool:
    return (
        bool(data.documents)
        and all(
            document.metadata.get(_SANITIZED_METADATA_KEY) is True for document in data.documents
        )
    ) or (not data.documents and SANITIZED_WARNING in data.report.warnings)


def validate_case_data(data: CaseData, *, sanitized: bool | None = None) -> None:
    is_sanitized = is_sanitized_case(data) if sanitized is None else sanitized
    sanitized_markers = tuple(
        document.metadata.get(_SANITIZED_METADATA_KEY) is True for document in data.documents
    )
    if not is_sanitized and (any(sanitized_markers) or SANITIZED_WARNING in data.report.warnings):
        raise IntegrityError("full case contains sanitized-data markers")
    if data.schema_version != SCHEMA_VERSION:
        raise CompatibilityError(f"unsupported case schema version: {data.schema_version!r}")
    if data.report.manifest.schema_version != SCHEMA_VERSION:
        raise CompatibilityError(
            f"unsupported manifest schema version: {data.report.manifest.schema_version!r}"
        )

    _require_unique_ids("source reference", data.source_references)
    _require_unique_ids("document", data.documents)
    _require_unique_ids("evidence span", data.evidence_spans)
    _require_unique_ids("mention", data.mentions)
    _require_unique_ids("claim", data.claims)
    _require_unique_ids("entity", data.entities)
    _require_unique_ids("event", data.events)
    _require_unique_ids("relationship", data.relationships)
    _require_unique_ids("review item", data.review_items)
    _require_unique_ids("build report review item", data.report.review_items)
    _require_unique_ids("review decision", data.review_decisions)

    source_by_id = {source.id: source for source in data.source_references}
    document_by_id = {document.id: document for document in data.documents}
    evidence_by_id = {span.id: span for span in data.evidence_spans}
    mention_by_id = {mention.id: mention for mention in data.mentions}
    claim_by_id = {claim.id: claim for claim in data.claims}
    entity_by_id = {entity.id: entity for entity in data.entities}
    event_by_id = {event.id: event for event in data.events}
    node_ids = set(entity_by_id) | set(event_by_id)
    review_target_ids = (
        set(source_by_id)
        | set(document_by_id)
        | set(evidence_by_id)
        | set(mention_by_id)
        | set(claim_by_id)
        | set(entity_by_id)
        | set(event_by_id)
        | {relationship.id for relationship in data.relationships}
    )

    for document in data.documents:
        if not document.source_reference_ids:
            raise IntegrityError(f"document {document.id} has no source references")
        _require_refs(
            f"document {document.id} source references",
            document.source_reference_ids,
            source_by_id,
        )
        for source_id in document.source_reference_ids:
            if source_by_id[source_id].content_sha256 != document.content_sha256:
                raise IntegrityError(
                    f"document {document.id} and source {source_id} have different content hashes"
                )
        if is_sanitized:
            if (
                document.raw_text
                or document.normalized_text
                or len(document.normalized_to_raw) != 1
                or document.normalized_to_raw[0] != 0
                or document.metadata.get(_SANITIZED_METADATA_KEY) is not True
            ):
                raise IntegrityError(f"sanitized document {document.id} retains full text")
        else:
            for segment in document.segments:
                if segment.raw_end > len(document.raw_text):
                    raise IntegrityError(
                        f"document {document.id} segment exceeds document raw text"
                    )
            if any(left > right for left, right in pairwise(document.normalized_to_raw)):
                raise IntegrityError(f"document {document.id} has a non-monotonic offset map")
        if not document.normalized_to_raw or document.normalized_to_raw[0] != 0:
            raise IntegrityError(f"document {document.id} offset map must start at raw boundary 0")
        if document.normalized_to_raw[-1] != len(document.raw_text):
            raise IntegrityError(
                f"document {document.id} offset map must end at the final raw boundary"
            )
        if any(
            boundary < 0 or boundary > len(document.raw_text)
            for boundary in document.normalized_to_raw
        ):
            raise IntegrityError(f"document {document.id} offset map has an invalid raw boundary")

    for span in data.evidence_spans:
        document = document_by_id.get(span.document_id)
        if document is None:
            raise IntegrityError(
                f"evidence span {span.id} references unknown document {span.document_id}"
            )
        if not is_sanitized:
            if span.raw_end > len(document.raw_text):
                raise IntegrityError(f"evidence span {span.id} exceeds document raw text")
            if document.raw_text[span.raw_start : span.raw_end] != span.quote:
                raise IntegrityError(f"evidence span {span.id} quote does not round-trip")
            if span.normalized_start is not None and span.normalized_end is not None:
                if span.normalized_end > len(document.normalized_text):
                    raise IntegrityError(f"evidence span {span.id} exceeds normalized text")
                mapped_start = document.normalized_to_raw[span.normalized_start]
                mapped_end = document.normalized_to_raw[span.normalized_end]
                if (mapped_start, mapped_end) != (span.raw_start, span.raw_end):
                    raise IntegrityError(
                        f"evidence span {span.id} normalized offsets do not map to raw offsets"
                    )

    for mention in data.mentions:
        if mention.evidence_span_id not in evidence_by_id:
            raise IntegrityError(
                f"mention {mention.id} references unknown evidence {mention.evidence_span_id}"
            )
    for claim in data.claims:
        _require_refs(f"claim {claim.id} evidence", claim.evidence_span_ids, evidence_by_id)
        _require_temporal_evidence(
            f"claim {claim.id} temporal expression",
            claim.temporal,
            evidence_by_id,
        )
        for participant in claim.participants:
            if participant.mention_id is not None and participant.mention_id not in mention_by_id:
                raise IntegrityError(
                    f"claim {claim.id} references unknown mention {participant.mention_id}"
                )
            if participant.entity_id is not None and participant.entity_id not in entity_by_id:
                raise IntegrityError(
                    f"claim {claim.id} references unknown entity {participant.entity_id}"
                )
    for entity in data.entities:
        _require_refs(f"entity {entity.id} mentions", entity.mention_ids, mention_by_id)
    for event in data.events:
        _require_refs(f"event {event.id} claims", event.claim_ids, claim_by_id)
        _require_refs(
            f"event {event.id} participants",
            event.participant_entity_ids,
            entity_by_id,
        )
        _require_temporal_evidence(
            f"event {event.id} temporal expression",
            event.temporal,
            evidence_by_id,
        )
    for relationship in data.relationships:
        if relationship.source_id not in node_ids or relationship.target_id not in node_ids:
            raise IntegrityError(f"relationship {relationship.id} references an unknown graph node")
        _require_refs(
            f"relationship {relationship.id} supporting claims",
            relationship.supporting_claim_ids,
            claim_by_id,
        )
        _require_refs(
            f"relationship {relationship.id} opposing claims",
            relationship.opposing_claim_ids,
            claim_by_id,
        )
    for item in (*data.review_items, *data.report.review_items):
        if not item.target_ids:
            raise IntegrityError(f"review item {item.id} has no targets")
        _require_refs(f"review item {item.id} evidence", item.evidence_span_ids, evidence_by_id)
        _require_refs(
            f"review item {item.id} targets",
            item.target_ids,
            review_target_ids,
        )

    for result in data.report.documents:
        if result.source_reference_id not in source_by_id:
            raise IntegrityError(
                f"build result references unknown source {result.source_reference_id}"
            )
        if result.document_id is not None and result.document_id not in document_by_id:
            raise IntegrityError(f"build result references unknown document {result.document_id}")
    try:
        ReviewEngine().effective_decisions(data.review_decisions)
    except ReviewDecisionError as error:
        raise IntegrityError("saved review decision log is invalid") from error


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise IntegrityError(
            "case contains a value that cannot be serialized canonically"
        ) from error
    except RecursionError as error:
        raise IntegrityError("case data exceeds the supported nesting depth") from error
    except MemoryError as error:
        raise PersistenceError("not enough memory to serialize case data safely") from error


def _validate_max_bytes(max_bytes: int) -> int:
    if isinstance(max_bytes, bool) or max_bytes < 1:
        raise PersistenceError("max_bytes must be a positive integer")
    return max_bytes


def _oversized_message(limit: int, *, source: Path | None = None) -> str:
    label = f"saved case {source}" if source is not None else "saved case"
    return f"{label} exceeds the configured serialized size limit of {limit} bytes"


def _enforce_json_nesting(text: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > MAX_JSON_NESTING_DEPTH:
                raise IntegrityError(
                    "saved case exceeds the supported JSON nesting depth "
                    f"of {MAX_JSON_NESTING_DEPTH}"
                )
        elif character in "]}":
            depth -= 1
            if depth < 0:
                # Let the JSON decoder report the general syntax error, but do
                # not allow malformed input to invalidate the depth accounting.
                depth = 0


def _enforce_record_budget(payload: Mapping[str, object]) -> None:
    count = 0

    def add_collection(value: object) -> list[object]:
        nonlocal count
        if not isinstance(value, list):
            return []
        items = cast("list[object]", value)
        count += len(items)
        if count > MAX_CASE_RECORDS:
            raise PersistenceError(
                f"saved case exceeds the aggregate record limit of {MAX_CASE_RECORDS}"
            )
        return items

    top_level_names = (
        "source_references",
        "documents",
        "evidence_spans",
        "mentions",
        "claims",
        "entities",
        "events",
        "relationships",
        "review_items",
        "review_decisions",
    )
    collections = {name: add_collection(payload.get(name)) for name in top_level_names}

    for document in collections["documents"]:
        if isinstance(document, dict):
            record = cast("dict[str, object]", document)
            add_collection(record.get("segments"))
    for evidence in collections["evidence_spans"]:
        if isinstance(evidence, dict):
            record = cast("dict[str, object]", evidence)
            add_collection(record.get("boxes"))
    for claim in collections["claims"]:
        if isinstance(claim, dict):
            record = cast("dict[str, object]", claim)
            add_collection(record.get("participants"))
            add_collection(record.get("temporal"))
    for event in collections["events"]:
        if isinstance(event, dict):
            record = cast("dict[str, object]", event)
            add_collection(record.get("temporal"))

    report = payload.get("report")
    if isinstance(report, dict):
        report_record = cast("dict[str, object]", report)
        results = add_collection(report_record.get("documents"))
        add_collection(report_record.get("failures"))
        add_collection(report_record.get("review_items"))
        for result in results:
            if isinstance(result, dict):
                result_record = cast("dict[str, object]", result)
                add_collection(result_record.get("failures"))


def _require_temporal_evidence(
    label: str,
    expressions: tuple[TemporalExpression, ...],
    evidence_by_id: Mapping[str, Any],
) -> None:
    for expression in expressions:
        evidence_span_id = expression.evidence_span_id
        if evidence_span_id is not None and evidence_span_id not in evidence_by_id:
            raise IntegrityError(f"{label} references unknown evidence {evidence_span_id}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise IntegrityError(f"saved case contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_non_finite(value: str) -> None:
    raise IntegrityError(f"saved case contains non-finite number {value}")


def _require_unique_ids(name: str, records: tuple[Any, ...]) -> None:
    seen: set[str] = set()
    for record in records:
        if record.id in seen:
            raise IntegrityError(f"duplicate {name} id: {record.id}")
        seen.add(record.id)


def _require_refs(
    label: str,
    references: tuple[str, ...],
    available: Collection[str],
) -> None:
    missing = sorted(set(references) - set(available))
    if missing:
        raise IntegrityError(f"{label} reference unknown ids: {', '.join(missing)}")
