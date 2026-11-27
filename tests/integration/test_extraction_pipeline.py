"""Milestone 1 -> 2 end to end: Docling parsing of the sample PDF/DOCX/XLSX, then
claim/entity extraction with a scripted model that answers from the prompt.

The scripted model reads the sources out of the real prompt (so source ids, row
rendering and context are exercised) and proposes claims a correct model should.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from factories import llm_claim, llm_entity, llm_json

from veritas.core.config import Settings
from veritas.core.contracts import Claim, Modality, Quantity
from veritas.extraction.pipeline import run_extraction
from veritas.extraction.repository import load_claims, load_entities, load_entity_documents
from veritas.ingestion.pipeline import IngestionPipeline
from veritas.llm.base import LLMRequest
from veritas.llm.mock_provider import MockProvider
from veritas.parsing.service import DocumentParser

pytestmark = pytest.mark.slow

_SOURCE = re.compile(r"\[(S\d+)\]\n(?:context: (.*)\n)?(text|row): (.*)")


def scripted_model(req: LLMRequest) -> str:
    claims, entities = [], []
    for sid, _ctx, _kind, text in _SOURCE.findall(req.messages[1].content):
        if "approved allocation of LKR 5 million" in text:
            claims.append(
                llm_claim(
                    source_id=sid,
                    valid_from="2025-01",
                    valid_to="2025-01",
                    time_expression="in January 2025",
                )
            )
        if "completed in June 2025" in text:
            claims.append(
                llm_claim(
                    source_id=sid,
                    predicate="completion_date",
                    object_type="date",
                    value="2025-06",
                    currency=None,
                    modality="unknown",
                    evidence_quotes=["completed in June 2025"],
                )
            )
        if "proposes a budget of LKR 7 million" in text:
            claims.append(
                llm_claim(
                    source_id=sid,
                    modality="proposed",
                    value="7000000",
                    valid_from="2024-11",
                    valid_to="2024-11",
                    time_expression="submitted in November 2024",
                    evidence_quotes=["proposes a budget of LKR 7 million"],
                )
            )
        if text.startswith("Project = Project Alpha | Modality = approved"):
            claims.append(
                llm_claim(
                    source_id=sid,
                    value="5500000",
                    valid_from="2025-01",
                    valid_to="2025-01",
                    time_expression="2025-01",
                    evidence_quotes=["5500000", "approved", "Project Alpha"],
                )
            )
        if "University of Sri Jayewardenepura (USJ)" in text:
            entities.append(
                llm_entity(
                    source_id=sid,
                    name="University of Sri Jayewardenepura",
                    entity_type="organization",
                    aliases=["USJ"],
                )
            )
        if "Sri Jayewardenepura University" in text:
            entities.append(
                llm_entity(
                    source_id=sid, name="Sri Jayewardenepura University", entity_type="organization"
                )
            )
        if "Project Alpha" in text:
            entities.append(llm_entity(source_id=sid, name="Project Alpha"))
    return llm_json(claims, entities)


@pytest.fixture(scope="module")
def parser() -> DocumentParser:
    return DocumentParser()


def test_samples_end_to_end(
    settings: Settings, sample_files: dict[str, Path], parser: DocumentParser
) -> None:
    pipe = IngestionPipeline(settings.model_copy(update={"max_upload_mb": 10}), parser)
    ids = {fmt: pipe.run(path).ingest.document.document_id for fmt, path in sample_files.items()}

    runs = run_extraction(
        settings,
        [ids["pdf"], ids["docx"], ids["xlsx"]],
        provider=MockProvider(responder=scripted_model),
        sessions=pipe.sessions,
    )
    assert [r.status for r in runs] == ["completed"] * 3

    def claims(fmt: str) -> list[Claim]:
        return load_claims(pipe.sessions, ids[fmt])

    # PDF: page provenance, approved modality, period
    budget = next(c for c in claims("pdf") if c.predicate == "budget")
    assert budget.modality is Modality.APPROVED
    assert budget.object == Quantity(value=5000000, currency="LKR")
    assert budget.evidence[0].location.page_number == 2
    assert budget.evidence[0].quote == "approved allocation of LKR 5 million"
    done = next(c for c in claims("pdf") if c.predicate == "completion_date")
    assert done.object == "2025-06" and done.evidence[0].location.page_number == 2

    # DOCX: proposed modality, structural location, no page
    (proposed,) = claims("docx")
    assert proposed.modality is Modality.PROPOSED
    loc = proposed.evidence[0].location
    assert loc.page_number is None and loc.heading_path[-1] == "Budget request"

    # XLSX: sheet/cell provenance for every quoted cell (value C2, modality B2,
    # subject A2) plus the period's own cell (D2, "2025-01"), which is added as
    # evidence because the period is taken from the same row.
    (row,) = claims("xlsx")
    data = {
        (e.location.sheet_name, e.location.cell_range) for e in row.evidence if e.role != "header"
    }
    assert data == {("Budget", "C2"), ("Budget", "B2"), ("Budget", "A2"), ("Budget", "D2")}
    # LKR comes from the value column's header cell, cited separately with its offsets.
    (hdr,) = [e for e in row.evidence if e.role == "header"]
    assert (hdr.location.sheet_name, hdr.location.cell_range, hdr.quote) == ("Budget", "C1", "LKR")
    assert row.valid_from == "2025-01"

    # The proposed (DOCX) and approved (PDF/XLSX) budgets remain separate claims.
    by_modality = {(c.predicate, c.modality) for f in ids for c in claims(f)}
    assert {("budget", Modality.PROPOSED), ("budget", Modality.APPROVED)} <= by_modality

    # Entities: USJ alias defined in the PDF; the DOCX's reordered name is not merged.
    entities = load_entities(pipe.sessions)
    by_id = {e.entity_id: e for e in entities}
    usj = next(e for e in entities if e.canonical_name == "University of Sri Jayewardenepura")
    assert "USJ" in usj.aliases
    assert "Sri Jayewardenepura University" in {e.canonical_name for e in entities}

    # "Project Alpha" is named in all three documents. Under the default
    # document-scoped policy each document has its own entity (same name, distinct
    # ids) and the cross-document identities wait in the review queue.
    alphas = [e for e in entities if e.normalized_name == "project alpha"]
    assert len(alphas) == 3 and len({e.entity_id for e in alphas}) == 3
    assert budget.subject_entity_id is not None
    assert by_id[budget.subject_entity_id].canonical_name == "Project Alpha"
    entity_docs = load_entity_documents(pipe.sessions)
    assert entity_docs[budget.subject_entity_id] == {ids["pdf"]}


def test_header_detection_on_real_docling_tables(
    settings: Settings, sample_files: dict[str, Path], parser: DocumentParser, tmp_path: Path
) -> None:
    """Docling 2.154 marks row 0 of every XLSX/DOCX table as column_header (even when
    it is data) and marks nothing in the sample PDF. The planner must not trust it."""
    from openpyxl import Workbook

    from veritas.core.contracts import DocumentFormat
    from veritas.extraction.sources import plan_sources
    from veritas.parsing.repository import load_spans

    headerless = tmp_path / "headerless.xlsx"
    wb = Workbook()
    ws = wb.active
    assert ws is not None
    ws.title = "Data"
    ws.append(["Project Alpha", 5500000, "2025-01"])
    ws.append(["Project Beta", 2500000, "2025-03"])
    wb.save(headerless)

    pipe = IngestionPipeline(settings.model_copy(update={"max_upload_mb": 10}), parser)

    def units(path: Path, fmt: DocumentFormat) -> list[str]:
        res = pipe.run(path)
        plan = plan_sources(load_spans(pipe.sessions, res.ingest.document.document_id), fmt)
        return [u.text for b in plan.batches for u in b if u.kind == "table_row"]

    # Headerless: the first data row is kept, no fabricated "Header = value" pairs.
    assert units(headerless, DocumentFormat.XLSX) == [
        "Project Alpha | 5500000 | 2025-01",
        "Project Beta | 2500000 | 2025-03",
    ]

    # Review reproduction, through Docling (which flags row 0 as a column header):
    # an uncertain first row is kept as data.
    repro = tmp_path / "repro.xlsx"
    wb2 = Workbook()
    ws2 = wb2.active
    assert ws2 is not None
    ws2.append(["Project Alpha", "pending"])
    ws2.append(["Project Beta", 5])
    wb2.save(repro)
    assert units(repro, DocumentFormat.XLSX) == ["Project Alpha | pending", "Project Beta | 5"]

    # Headerless DOCX table (Docling also flags its row 0 as a header).
    from docx import Document as DocxDocument

    docx_path = tmp_path / "headerless.docx"
    d = DocxDocument()
    t = d.add_table(rows=2, cols=2)
    for r, (a, b) in enumerate([("Project Alpha", "pending"), ("Project Beta", "5")]):
        t.cell(r, 0).text, t.cell(r, 1).text = a, b
    d.save(str(docx_path))
    assert units(docx_path, DocumentFormat.DOCX) == ["Project Alpha | pending", "Project Beta | 5"]
    # Headered tables in all three formats are still detected.
    assert "Amount (LKR) = 5500000" in " ".join(units(sample_files["xlsx"], DocumentFormat.XLSX))
    assert "Proposed cost (LKR) = 4,000,000" in " ".join(
        units(sample_files["docx"], DocumentFormat.DOCX)
    )
    assert "Approved budget (LKR) = 5,000,000" in " ".join(
        units(sample_files["pdf"], DocumentFormat.PDF)
    )
