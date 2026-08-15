from __future__ import annotations

from pathlib import Path

from docchrono import Case
from docchrono.cli import main
from docchrono.domain import EntityType

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "mini_case"


def test_case_build_save_load_and_cli(tmp_path: Path) -> None:
    case = Case.build(FIXTURE_DIR)

    assert case.report.complete
    assert len(case.documents) == 3
    assert case.entities
    assert case.events
    assert case.relationships
    assert tuple(case.timeline)
    paid_event = next(event for event in case.events if event.title.startswith("Paid:"))
    assert case.graph.neighbors(paid_event)
    assert case.graph.find_path("Acme Corporation", "Contract #428") is not None
    assert not any(
        entity.type == EntityType.PERSON and entity.canonical_name == "On March"
        for entity in case.entities
    )
    assert any(
        entity.type == EntityType.ORGANIZATION and entity.canonical_name == "Acme Corporation"
        for entity in case.entities
    )
    assert not any(
        entity.type == EntityType.PERSON and entity.canonical_name == "Acme Corporation"
        for entity in case.entities
    )

    document_by_id = {document.id: document for document in case.documents}
    for span in case.evidence_spans:
        document = document_by_id[span.document_id]
        assert document.raw_text[span.raw_start : span.raw_end] == span.quote
        assert span.normalized_start is not None
        assert span.normalized_end is not None
        assert document.normalized_to_raw[span.normalized_start] == span.raw_start
        assert document.normalized_to_raw[span.normalized_end] == span.raw_end

    first_path = tmp_path / "first.case.json"
    second_path = tmp_path / "second.case.json"
    case.save(first_path)
    case.save(second_path)
    assert first_path.read_bytes() == second_path.read_bytes()
    assert Case.load(first_path).data == case.data

    cli_path = tmp_path / "cli.case.json"
    assert main(["build", str(FIXTURE_DIR), "--output", str(cli_path), "--json"]) == 0
    assert cli_path.exists()
