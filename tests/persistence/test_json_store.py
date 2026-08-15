import hashlib
import json
from pathlib import Path

import pytest

from docchrono.case import Case
from docchrono.domain import (
    BuildManifest,
    BuildReport,
    BuildStage,
    CaseConfig,
    CaseData,
    Claim,
    ClaimKind,
    ClaimParticipant,
    Document,
    DocumentBuildResult,
    DocumentSegment,
    Entity,
    EntityType,
    Event,
    EventType,
    EvidenceSpan,
    Mention,
    OffsetMap,
    Relationship,
    ReviewItem,
    SourceReference,
    TemporalExpression,
)
from docchrono.errors import IntegrityError, PersistenceError
from docchrono.persistence import (
    MAX_CASE_RECORDS,
    MAX_JSON_NESTING_DEPTH,
    SANITIZED_WARNING,
    SanitizedCaseWarning,
    dumps_case_data,
    json_store,
    load_case_data,
    loads_case_data,
    save_case_data,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _envelope(payload: object) -> bytes:
    payload_bytes = _canonical(payload)
    return _canonical(
        {
            "format": "docchrono.case",
            "format_version": 1,
            "payload": payload,
            "payload_sha256": hashlib.sha256(payload_bytes).hexdigest(),
            "sanitized": False,
            "schema_version": "1.0",
        }
    )


def _data() -> CaseData:
    raw_text = "Alice approved."
    digest = "a" * 64
    source = SourceReference(
        id="source-a",
        path="case/a.txt",
        filename="a.txt",
        media_type="text/plain",
        size_bytes=len(raw_text.encode()),
        content_sha256=digest,
    )
    document = Document(
        id="document-a",
        content_sha256=digest,
        source_reference_ids=(source.id,),
        media_type="text/plain",
        raw_text=raw_text,
        normalized_text=raw_text,
        normalized_to_raw=OffsetMap.identity(len(raw_text)),
        parser_name="fixture",
        parser_version="1",
    )
    evidence = EvidenceSpan(
        id="evidence-a",
        document_id=document.id,
        raw_start=0,
        raw_end=5,
        normalized_start=0,
        normalized_end=5,
        quote="Alice",
    )
    mention = Mention(
        id="mention-a",
        evidence_span_id=evidence.id,
        entity_type=EntityType.PERSON,
        text="Alice",
        normalized_text="alice",
        extractor_name="fixture",
        extractor_version="1",
        score=0.9,
    )
    entity = Entity(
        id="entity-a",
        type=EntityType.PERSON,
        canonical_name="Alice",
        aliases=(),
        mention_ids=(mention.id,),
        score=0.9,
    )
    claim = Claim(
        id="claim-a",
        kind=ClaimKind.EVENT,
        predicate="APPROVED",
        participants=(ClaimParticipant(role="actor", entity_id=entity.id),),
        evidence_span_ids=(evidence.id,),
        extractor_name="fixture",
        extractor_version="1",
        score=0.9,
    )
    event = Event(
        id="event-a",
        type=EventType.ACTION,
        title="Approval",
        claim_ids=(claim.id,),
        participant_entity_ids=(entity.id,),
        score=0.9,
    )
    relationship = Relationship(
        id="relationship-a",
        source_id=entity.id,
        target_id=event.id,
        type="PARTICIPATED_IN",
        supporting_claim_ids=(claim.id,),
        score=0.9,
    )
    manifest = BuildManifest(
        docchrono_version="0.1.0",
        python_version="3.13",
        platform="test",
        config_fingerprint="fixture",
    )
    report = BuildReport(
        requested_stage=BuildStage.GRAPH,
        completed_stage=BuildStage.GRAPH,
        documents=(
            DocumentBuildResult(
                source_reference_id=source.id,
                document_id=document.id,
                completed_stage=BuildStage.GRAPH,
            ),
        ),
        manifest=manifest,
    )
    return CaseData(
        config=CaseConfig(),
        source_references=(source,),
        documents=(document,),
        evidence_spans=(evidence,),
        mentions=(mention,),
        claims=(claim,),
        entities=(entity,),
        events=(event,),
        relationships=(relationship,),
        report=report,
    )


def test_canonical_bytes_and_atomic_save_load_round_trip(tmp_path: Path) -> None:
    data = _data()

    first = dumps_case_data(data)
    second = dumps_case_data(data)
    target = tmp_path / "case.json"
    assert save_case_data(data, target) == target

    assert first == second == target.read_bytes()
    assert load_case_data(target) == data
    assert not tuple(tmp_path.glob("*.tmp"))


def test_payload_corruption_is_detected() -> None:
    encoded = dumps_case_data(_data())
    envelope = json.loads(encoded)
    envelope["payload"]["entities"][0]["canonical_name"] = "Mallory"
    corrupted = json.dumps(envelope).encode()

    with pytest.raises(IntegrityError, match="hash does not match"):
        loads_case_data(corrupted)


def test_sanitized_save_omits_document_text_but_retains_quoted_evidence() -> None:
    data = _data()

    with pytest.warns(SanitizedCaseWarning, match="evidence-span quotes remain"):
        encoded = dumps_case_data(data, sanitized=True)
    restored = loads_case_data(encoded)

    assert restored.documents[0].raw_text == ""
    assert restored.documents[0].normalized_text == ""
    assert restored.evidence_spans[0].quote == "Alice"
    assert SANITIZED_WARNING in restored.report.warnings


def test_invalid_evidence_quote_is_rejected_before_save() -> None:
    data = _data()
    invalid_span = data.evidence_spans[0].model_copy(update={"quote": "Wrong"})
    invalid = data.model_copy(update={"evidence_spans": (invalid_span,)})

    with pytest.raises(IntegrityError, match="does not round-trip"):
        dumps_case_data(invalid)


def test_normalized_evidence_boundaries_must_map_to_raw_boundaries() -> None:
    data = _data()
    invalid_span = data.evidence_spans[0].model_copy(update={"normalized_start": 1})
    invalid = data.model_copy(update={"evidence_spans": (invalid_span,)})

    with pytest.raises(IntegrityError, match="normalized offsets do not map"):
        dumps_case_data(invalid)


def test_in_memory_and_file_loads_enforce_byte_limit_with_explicit_override(
    tmp_path: Path,
) -> None:
    encoded = dumps_case_data(_data())
    target = tmp_path / "case.json"
    target.write_bytes(encoded)

    with pytest.raises(PersistenceError, match="serialized size limit"):
        loads_case_data(encoded, max_bytes=len(encoded) - 1)
    with pytest.raises(PersistenceError, match="serialized size limit"):
        load_case_data(target, max_bytes=len(encoded) - 1)

    assert loads_case_data(encoded, max_bytes=len(encoded)) == _data()
    assert load_case_data(target, max_bytes=len(encoded)) == _data()
    assert Case.load(target, max_bytes=len(encoded)).data == _data()


def test_invalid_load_limit_is_normalized_as_persistence_error() -> None:
    with pytest.raises(PersistenceError, match="positive integer"):
        loads_case_data(b"{}", max_bytes=0)


def test_json_nesting_is_rejected_before_decoder_recursion() -> None:
    nested = ("[" * (MAX_JSON_NESTING_DEPTH + 1)) + ("]" * (MAX_JSON_NESTING_DEPTH + 1))

    with pytest.raises(IntegrityError, match="nesting depth"):
        loads_case_data(nested)


def test_brackets_inside_json_strings_do_not_count_as_nesting() -> None:
    encoded = dumps_case_data(_data())
    text = encoded.decode("utf-8").replace(
        '"platform":"test"',
        '"platform":"[[[[{{{{test}}}}]]]]"',
    )
    envelope = json.loads(text)
    envelope["payload_sha256"] = hashlib.sha256(_canonical(envelope["payload"])).hexdigest()

    restored = loads_case_data(_canonical(envelope))

    assert restored.report.manifest.platform == "[[[[{{{{test}}}}]]]]"


def test_aggregate_record_budget_is_enforced_before_schema_validation() -> None:
    source_records: list[object] = [{} for _ in range(MAX_CASE_RECORDS + 1)]
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "source_references": source_records,
    }

    with pytest.raises(PersistenceError, match="aggregate record limit"):
        loads_case_data(_envelope(payload))


@pytest.mark.parametrize(
    ("failure", "expected_error", "message"),
    [
        (RecursionError("deep"), IntegrityError, "nesting depth"),
        (MemoryError("large"), PersistenceError, "memory"),
    ],
)
def test_decoder_resource_failures_are_normalized(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    expected_error: type[Exception],
    message: str,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> object:
        raise failure

    monkeypatch.setattr(json_store.json, "loads", fail)

    with pytest.raises(expected_error, match=message):
        loads_case_data("{}")


@pytest.mark.parametrize(
    ("failure", "expected_error", "message"),
    [
        (RecursionError("deep"), IntegrityError, "nesting depth"),
        (MemoryError("large"), PersistenceError, "memory"),
    ],
)
def test_serializer_resource_failures_are_normalized(
    monkeypatch: pytest.MonkeyPatch,
    failure: BaseException,
    expected_error: type[Exception],
    message: str,
) -> None:
    def fail(*_args: object, **_kwargs: object) -> str:
        raise failure

    monkeypatch.setattr(json_store.json, "dumps", fail)

    with pytest.raises(expected_error, match=message):
        dumps_case_data(_data())


def test_document_segments_must_stay_within_raw_text() -> None:
    data = _data()
    invalid_document = data.documents[0].model_copy(
        update={"segments": (DocumentSegment(kind="paragraph", raw_start=0, raw_end=10_000),)}
    )
    invalid = data.model_copy(update={"documents": (invalid_document,)})

    with pytest.raises(IntegrityError, match="segment exceeds"):
        dumps_case_data(invalid)


@pytest.mark.parametrize(
    ("offset_map", "message"),
    [
        (tuple(range(1, 17)), "start at raw boundary 0"),
        ((*range(15), 14), "end at the final raw boundary"),
    ],
)
def test_document_offset_map_requires_complete_raw_boundaries(
    offset_map: tuple[int, ...],
    message: str,
) -> None:
    data = _data()
    invalid_document = data.documents[0].model_copy(update={"normalized_to_raw": offset_map})
    invalid = data.model_copy(update={"documents": (invalid_document,)})

    with pytest.raises(IntegrityError, match=message):
        dumps_case_data(invalid)


@pytest.mark.parametrize("owner", ["claim", "event"])
def test_temporal_evidence_must_reference_an_actual_span(owner: str) -> None:
    data = _data()
    temporal = (TemporalExpression(original_text="yesterday", evidence_span_id="missing-evidence"),)
    if owner == "claim":
        invalid = data.model_copy(
            update={"claims": (data.claims[0].model_copy(update={"temporal": temporal}),)}
        )
    else:
        invalid = data.model_copy(
            update={"events": (data.events[0].model_copy(update={"temporal": temporal}),)}
        )

    with pytest.raises(IntegrityError, match="temporal expression references unknown evidence"):
        dumps_case_data(invalid)


def test_review_item_targets_must_reference_actual_records() -> None:
    data = _data()
    invalid_item = ReviewItem(
        id="review-missing",
        kind="claim_candidate",
        target_ids=("missing-claim",),
        reason="adversarial fixture",
    )
    invalid = data.model_copy(update={"review_items": (invalid_item,)})

    with pytest.raises(IntegrityError, match="targets reference unknown ids"):
        dumps_case_data(invalid)


def test_full_case_cannot_contain_partial_sanitization_markers() -> None:
    data = _data()
    sanitized_document = data.documents[0].model_copy(
        update={
            "id": "document-b",
            "raw_text": "",
            "normalized_text": "",
            "normalized_to_raw": OffsetMap.identity(0),
            "metadata": {"docchrono_sanitized_text": True},
        }
    )
    invalid = data.model_copy(update={"documents": (*data.documents, sanitized_document)})

    with pytest.raises(IntegrityError, match="sanitized-data markers"):
        dumps_case_data(invalid)


def test_sanitized_marker_cannot_be_downgraded_to_full_case() -> None:
    with pytest.warns(SanitizedCaseWarning):
        envelope = json.loads(dumps_case_data(_data(), sanitized=True))
    envelope["sanitized"] = False

    with pytest.raises(IntegrityError, match="sanitization marker"):
        loads_case_data(_canonical(envelope))
