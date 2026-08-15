from __future__ import annotations

import os
from pathlib import Path

import pytest

from docchrono.domain import CaseConfig, DocumentFailureCode
from docchrono.ingestion import Ingestor, LoadedDocument, LoaderRegistry, LoadRequest
from docchrono.ingestion import ingestor as ingestor_module


def test_ingestor_discovers_stably_and_deduplicates_by_bytes(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "z.txt").write_bytes(b"Same\r\ntext.")
    (tmp_path / "a.txt").write_bytes(b"Same\r\ntext.")
    (tmp_path / "b.md").write_text("# Heading\n\nDifferent.", encoding="utf-8")
    (tmp_path / "ignored.bin").write_bytes(b"unsupported")

    first = Ingestor().ingest(tmp_path)
    second = Ingestor().ingest(tmp_path)

    assert len(first.source_references) == 4
    assert len(first.documents) == 2
    assert [source.path for source in first.source_references] == sorted(
        (source.path for source in first.source_references),
        key=lambda value: (value.casefold(), value),
    )
    duplicate = next(
        document for document in first.documents if document.raw_text.startswith("Same")
    )
    assert len(duplicate.source_reference_ids) == 2
    assert duplicate.normalized_text == "Same\ntext."
    assert [failure.code for failure in first.failures] == [DocumentFailureCode.UNSUPPORTED_FORMAT]
    assert first == second


def test_duplicate_with_unsupported_suffix_links_to_supported_document(tmp_path: Path) -> None:
    (tmp_path / "copy.bin").write_bytes(b"same")
    (tmp_path / "source.txt").write_bytes(b"same")

    result = Ingestor().ingest(tmp_path)

    assert len(result.documents) == 1
    assert len(result.documents[0].source_reference_ids) == 2
    assert not result.failures
    assert "unsupported-suffix" in result.warnings[0]


def test_oversized_source_is_rejected_without_opening_or_false_content_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "large.txt"
    source.write_bytes(b"12345")
    open_calls = 0

    def unexpected_open(_path: object, _flags: int) -> int:
        nonlocal open_calls
        open_calls += 1
        raise AssertionError("oversized file must not be opened")

    monkeypatch.setattr(ingestor_module.os, "open", unexpected_open)

    result = Ingestor().ingest(source, CaseConfig(max_file_bytes=4))

    assert result.documents == ()
    assert result.source_references == ()
    assert result.failures[0].code is DocumentFailureCode.LIMIT_EXCEEDED
    assert open_calls == 0


def test_file_count_budget_is_deterministic(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("second", encoding="utf-8")
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")

    result = Ingestor().ingest(tmp_path, CaseConfig(max_files=1))

    assert [source.filename for source in result.source_references] == ["a.txt"]
    assert [document.raw_text for document in result.documents] == ["first"]
    assert result.failures[0].code is DocumentFailureCode.LIMIT_EXCEEDED
    assert "file count" in result.failures[0].message


def test_total_byte_budget_stops_before_exceeding_file(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_bytes(b"12")
    (tmp_path / "b.txt").write_bytes(b"34")

    result = Ingestor().ingest(tmp_path, CaseConfig(max_total_bytes=3))

    assert [source.filename for source in result.source_references] == ["a.txt"]
    assert result.source_references[0].size_bytes == 2
    assert result.failures[0].code is DocumentFailureCode.LIMIT_EXCEEDED
    assert "collection limit" in result.failures[0].message


def test_depth_budget_skips_deep_branch_but_keeps_shallow_files(tmp_path: Path) -> None:
    deep = tmp_path / "a-deep" / "nested"
    deep.mkdir(parents=True)
    (deep / "hidden.txt").write_text("hidden", encoding="utf-8")
    (tmp_path / "root.txt").write_text("visible", encoding="utf-8")

    result = Ingestor().ingest(tmp_path, CaseConfig(max_depth=1))

    assert [source.filename for source in result.source_references] == ["root.txt"]
    assert result.documents[0].raw_text == "visible"
    assert result.failures[0].code is DocumentFailureCode.LIMIT_EXCEEDED
    assert "depth" in result.failures[0].message


def test_source_mutation_during_open_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.txt"
    source.write_bytes(b"before")
    real_open = ingestor_module.os.open

    def mutate_then_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes], flags: int
    ) -> int:
        source.write_bytes(b"changed")
        return real_open(path, flags)

    monkeypatch.setattr(ingestor_module.os, "open", mutate_then_open)

    result = Ingestor().ingest(source)

    assert result.documents == ()
    assert result.failures[0].code is DocumentFailureCode.SOURCE_UNAVAILABLE
    assert "changed" in result.failures[0].message


def test_general_extracted_character_limit_applies_to_custom_loaders(tmp_path: Path) -> None:
    class ExpandingLoader:
        name = "expanding"
        version = "1"
        supported_suffixes = (".expand",)
        media_type = "text/plain"

        def load(self, request: LoadRequest) -> LoadedDocument:
            return LoadedDocument(raw_text="expanded", media_type=self.media_type)

    source = tmp_path / "source.expand"
    source.write_bytes(b"x")

    result = Ingestor(LoaderRegistry((ExpandingLoader(),))).ingest(
        source, CaseConfig(max_extracted_chars=4)
    )

    assert result.documents == ()
    assert result.failures[0].code is DocumentFailureCode.LIMIT_EXCEEDED


def test_missing_source_is_a_typed_discovery_failure(tmp_path: Path) -> None:
    result = Ingestor().ingest(tmp_path / "missing")

    assert result.source_references == ()
    assert result.failures[0].code is DocumentFailureCode.SOURCE_UNAVAILABLE
    assert result.failures[0].source_reference_id is None


def test_symlink_is_not_followed_by_default(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("target", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("creating symlinks is not permitted on this platform")

    result = Ingestor().ingest(link)

    assert result.source_references == ()
    assert result.documents == ()
    assert result.warnings and "Skipped symlink" in result.warnings[0]


def test_followed_symlink_cannot_escape_requested_directory(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = root / "escape.txt"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("creating symlinks is not permitted on this platform")

    result = Ingestor().ingest(root, CaseConfig(follow_symlinks=True))

    assert result.documents == ()
    assert result.failures[0].code is DocumentFailureCode.SOURCE_UNAVAILABLE
    assert "outside requested root" in result.failures[0].message


def test_adapter_output_contract_failure_is_typed(tmp_path: Path) -> None:
    class WrongMediaLoader:
        name = "wrong-media"
        version = "1"
        supported_suffixes = (".wrong",)
        media_type = "text/correct"

        def load(self, request: LoadRequest) -> LoadedDocument:
            return LoadedDocument(raw_text="text", media_type="text/wrong")

    source = tmp_path / "source.wrong"
    source.write_text("text", encoding="utf-8")

    result = Ingestor(LoaderRegistry((WrongMediaLoader(),))).ingest(source)

    assert result.failures[0].code is DocumentFailureCode.ADAPTER_CONTRACT


@pytest.mark.skipif(os.name != "nt", reason="Windows-specific path case behavior")
def test_repeated_input_path_is_only_ingested_once(tmp_path: Path) -> None:
    source = tmp_path / "One.txt"
    source.write_text("one", encoding="utf-8")

    result = Ingestor().ingest((source, source))

    assert len(result.source_references) == 1
    assert len(result.documents) == 1
