"""Parsing and end-to-end ingestion against real Docling.

The first run downloads Docling's layout models (~5 min on the reference
machine); later runs take ~15 s. Skip with:  pytest -m "not slow"
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from docx import Document as DocxDocument
from openpyxl import Workbook
from reportlab.pdfgen import canvas

from veritas.core.config import Settings
from veritas.core.contracts import DocumentSpan, DocumentStatus, Evidence, SpanKind
from veritas.ingestion.pipeline import IngestionPipeline
from veritas.parsing.repository import load_spans
from veritas.parsing.service import DocumentParser, ParsedDocument

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def parser() -> DocumentParser:
    return DocumentParser()  # one converter for the module: model loading is slow


@pytest.fixture
def pipeline(settings: Settings, parser: DocumentParser) -> IngestionPipeline:
    return IngestionPipeline(settings.model_copy(update={"max_upload_mb": 10}), parser)


def run(pipeline: IngestionPipeline, path: Path) -> ParsedDocument:
    res = pipeline.run(path)
    assert res.export_path is not None
    return ParsedDocument.model_validate_json(res.export_path.read_text())


def find(spans: list[DocumentSpan], text: str, kind: SpanKind | None = None) -> DocumentSpan:
    for s in spans:
        if text in s.text and (kind is None or s.kind is kind):
            return s
    raise AssertionError(f"no span containing {text!r}")


# ------------------------------------------------------------------- PDF


def test_pdf(pipeline: IngestionPipeline, sample_files: dict[str, Path]) -> None:
    parsed = run(pipeline, sample_files["pdf"])
    r = parsed.report
    assert r.status is DocumentStatus.PARSED
    assert r.total_pages == 2
    assert r.tables_extracted == 1 and r.headings_extracted >= 2
    assert r.provenance_available and r.spans_missing_provenance == 0

    sentence = find(parsed.spans, "approved allocation of LKR 5 million", SpanKind.TEXT)
    assert sentence.location is not None and sentence.location.page_number == 2
    assert "2. Project Alpha" in sentence.location.heading_path

    cell = find(parsed.spans, "5,000,000", SpanKind.TABLE_CELL)
    assert cell.location is not None
    assert cell.location.page_number == 2
    assert (cell.location.table_row, cell.location.table_col) == (1, 1)
    assert cell.parent_span_id == f"{r.document_id}:#/tables/0"

    ev = Evidence.from_span("EV-1", sentence, "LKR 5 million")
    assert ev.location.page_number == 2


# ------------------------------------------------------------------ DOCX


def test_docx(pipeline: IngestionPipeline, sample_files: dict[str, Path]) -> None:
    parsed = run(pipeline, sample_files["docx"])
    r = parsed.report
    assert r.status is DocumentStatus.PARSED
    assert r.total_pages is None  # Docling has no page model for DOCX; none invented
    assert r.provenance_available

    para = find(parsed.spans, "LKR 7 million", SpanKind.TEXT)
    assert para.location is not None
    assert para.location.page_number is None
    assert para.location.element_id.startswith("#/texts/")
    assert para.location.heading_path == ["Project Alpha — Funding Proposal", "Budget request"]

    title = find(parsed.spans, "Funding Proposal", SpanKind.TITLE)
    assert title.location is not None and title.location.heading_path == []

    cell = find(parsed.spans, "4,000,000", SpanKind.TABLE_CELL)
    assert cell.location is not None
    assert (cell.location.table_row, cell.location.table_col) == (1, 1)


# ------------------------------------------------------------------ XLSX


def test_xlsx(pipeline: IngestionPipeline, sample_files: dict[str, Path]) -> None:
    parsed = run(pipeline, sample_files["xlsx"])
    r = parsed.report
    assert r.status is DocumentStatus.PARSED
    assert r.total_sheets == 2 and r.tables_extracted == 2
    assert r.provenance_available

    budget = find(parsed.spans, "5500000", SpanKind.TABLE_CELL)
    assert budget.location is not None
    assert (budget.location.sheet_name, budget.location.cell_range) == ("Budget", "C2")

    # "2025-09" appears on both sheets; each must resolve to its own sheet and cell.
    located = {
        (s.location.sheet_name, s.location.cell_range)
        for s in parsed.spans
        if s.kind is SpanKind.TABLE_CELL and s.text == "2025-09" and s.location is not None
    }
    assert located == {("Budget", "D3"), ("Milestones", "C2")}

    table = next(s for s in parsed.spans if s.kind is SpanKind.TABLE)
    assert table.location is not None and table.location.cell_range == "A1:D4"


def test_xlsx_table_not_at_a1(pipeline: IngestionPipeline, tmp_path: Path) -> None:
    p = tmp_path / "offset.xlsx"
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Offset"
    ws["C3"], ws["D3"], ws["C4"], ws["D4"] = "Item", "Amount", "Alpha", 42
    wb.save(p)

    spans = run(pipeline, p).spans
    for text, ref in [("Item", "C3"), ("Amount", "D3"), ("Alpha", "C4"), ("42", "D4")]:
        loc = find(spans, text, SpanKind.TABLE_CELL).location
        assert loc is not None and (loc.sheet_name, loc.cell_range) == ("Offset", ref)


# --------------------------------------------------- corrupted / empty


def test_corrupted_pdf_fails_cleanly(
    pipeline: IngestionPipeline, sample_files: dict[str, Path], tmp_path: Path
) -> None:
    """Valid %PDF- header (passes validation) but truncated body."""
    data = sample_files["pdf"].read_bytes()
    p = tmp_path / "truncated.pdf"
    p.write_bytes(data[: len(data) // 3])

    res = pipeline.run(p)
    assert res.report is not None
    assert res.report.status is DocumentStatus.FAILED
    assert res.report.errors
    assert pipeline.ingestion.get(res.ingest.document.document_id).status == "failed"  # type: ignore[union-attr]


def test_empty_documents(pipeline: IngestionPipeline, tmp_path: Path) -> None:
    pdf = tmp_path / "blank.pdf"
    c = canvas.Canvas(str(pdf))
    c.showPage()
    c.save()
    docx = tmp_path / "blank.docx"
    DocxDocument().save(str(docx))
    xlsx = tmp_path / "blank.xlsx"
    Workbook().save(xlsx)

    for p in (pdf, docx, xlsx):
        parsed = run(pipeline, p)
        assert parsed.report.status is DocumentStatus.PARSED, p.name
        assert parsed.spans == [], p.name
        assert not parsed.report.provenance_available
        assert any("no text elements" in w for w in parsed.report.warnings), p.name


# ------------------------------------------------------- persistence


def test_spans_persisted_and_exported(
    pipeline: IngestionPipeline, sample_files: dict[str, Path], settings: Settings
) -> None:
    res = pipeline.run(sample_files["xlsx"])
    assert res.report is not None and res.export_path is not None
    doc_id = res.ingest.document.document_id

    exported = ParsedDocument.model_validate_json(res.export_path.read_text())
    stored = load_spans(pipeline.sessions, doc_id)
    assert stored == exported.spans
    assert json.loads((settings.parsed_dir / f"{doc_id}.docling.json").read_text())

    doc = pipeline.ingestion.get(doc_id)
    assert doc is not None and doc.status == "parsed"


def test_duplicate_is_not_reparsed(
    pipeline: IngestionPipeline, sample_files: dict[str, Path]
) -> None:
    first = pipeline.run(sample_files["docx"])
    second = pipeline.run(sample_files["docx"])
    assert not first.ingest.duplicate and second.ingest.duplicate
    assert second.report is None
    third = pipeline.run(sample_files["docx"], reparse_duplicates=True)
    assert third.report is not None
    # Re-parsing replaces spans rather than duplicating them.
    assert len(load_spans(pipeline.sessions, first.ingest.document.document_id)) == len(
        ParsedDocument.model_validate_json(third.export_path.read_text()).spans  # type: ignore[union-attr]
    )
