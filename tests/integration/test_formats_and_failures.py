from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document as WordDocument

from docchrono import BuildFailed, Case, SourceError
from docchrono.domain import DocumentFailureCode


def _write_minimal_pdf(path: Path, text: str) -> None:
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream = f"BT /F1 12 Tf 72 720 Td ({escaped}) Tj ET".encode("ascii")
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length "
        + str(len(stream)).encode("ascii")
        + b" >>\nstream\n"
        + stream
        + b"\nendstream",
    )
    content = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for number, body in enumerate(objects, start=1):
        offsets.append(len(content))
        content.extend(f"{number} 0 obj\n".encode("ascii"))
        content.extend(body)
        content.extend(b"\nendobj\n")
    xref_offset = len(content)
    content.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    content.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        content.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    content.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )
    path.write_bytes(bytes(content))


def test_case_builds_docx_and_text_pdf_with_provenance(tmp_path: Path) -> None:
    authored = WordDocument()
    authored.add_paragraph("Alice Smith approved Invoice #381 on August 12, 2026.")
    authored.save(str(tmp_path / "approval.docx"))
    _write_minimal_pdf(
        tmp_path / "payment.pdf",
        "Robert Williams paid Invoice #381 on August 13, 2026.",
    )

    case = Case.build(tmp_path)

    assert case.report.complete
    assert {document.media_type for document in case.documents} == {
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    }
    assert {document.parser_name for document in case.documents} == {
        "pdfplumber",
        "python-docx",
    }
    assert case.events
    assert any(span.page == 1 for span in case.evidence_spans)
    pdf_ids = {
        document.id for document in case.documents if document.media_type == "application/pdf"
    }
    assert any(span.boxes for span in case.evidence_spans if span.document_id in pdf_ids)


def test_non_strict_build_retains_typed_failure_and_strict_build_raises(
    tmp_path: Path,
) -> None:
    (tmp_path / "supported.txt").write_text(
        "Alice Smith approved Invoice #381 on August 12, 2026.",
        encoding="utf-8",
    )
    (tmp_path / "unsupported.csv").write_text("name,value\nAlice,1\n", encoding="utf-8")

    case = Case.build(tmp_path)

    assert not case.report.complete
    failures = tuple(failure for result in case.report.documents for failure in result.failures)
    assert [failure.code for failure in failures] == [DocumentFailureCode.UNSUPPORTED_FORMAT]

    with pytest.raises(BuildFailed) as failure_info:
        Case.build(tmp_path, strict=True)
    assert not failure_info.value.report.complete


def test_missing_source_raises_source_error(tmp_path: Path) -> None:
    with pytest.raises(SourceError):
        Case.build(tmp_path / "does-not-exist")
