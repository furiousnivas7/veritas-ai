"""Builders for extraction tests: spans, seeded documents and model responses.

No Docling involved. Spans are constructed directly with the same location rules
that the parser produces (checked by tests/integration/test_parsing.py).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from veritas.core.contracts import (
    DocumentFormat,
    DocumentSpan,
    DocumentStatus,
    SourceLocation,
    SpanKind,
)
from veritas.core.db import DocumentRow
from veritas.parsing.repository import save_parse
from veritas.parsing.service import ParsedDocument, ParseReport


def span(
    doc_id: str,
    element_id: str,
    text: str,
    *,
    fmt: DocumentFormat = DocumentFormat.PDF,
    kind: SpanKind = SpanKind.TEXT,
    order: int = 0,
    page: int | None = 1,
    heading_path: list[str] | None = None,
    sheet: str | None = None,
    cell: str | None = None,
    table: tuple[str, int, int] | None = None,
    located: bool = True,
) -> DocumentSpan:
    suffix = f":r{table[1]}c{table[2]}" if table else ""
    loc = None
    if located:
        loc = SourceLocation(
            format=fmt,
            element_id=element_id,
            page_number=page if fmt is DocumentFormat.PDF else None,
            sheet_name=sheet,
            cell_range=cell,
            heading_path=heading_path or [],
            table_ref=table[0] if table else None,
            table_row=table[1] if table else None,
            table_col=table[2] if table else None,
        )
    return DocumentSpan(
        span_id=f"{doc_id}:{element_id}{suffix}",
        document_id=doc_id,
        document_version=1,
        kind=kind,
        text=text,
        order=order,
        location=loc,
        missing_provenance=None if located else "docling exposed no page provenance",
    )


def xlsx_table(
    doc_id: str,
    sheet: str,
    rows: list[list[str]],
    ref: str = "#/tables/0",
    origin_col: int = 0,
    origin_row: int = 0,
) -> list[DocumentSpan]:
    """Cell spans for a sheet table whose top-left cell is (origin_col, origin_row)."""
    out = []
    for r, row in enumerate(rows):
        for c, text in enumerate(row):
            col_letter = chr(ord("A") + origin_col + c)
            out.append(
                span(
                    doc_id,
                    ref,
                    text,
                    fmt=DocumentFormat.XLSX,
                    kind=SpanKind.TABLE_CELL,
                    order=len(out),
                    page=None,
                    sheet=sheet,
                    cell=f"{col_letter}{origin_row + r + 1}",
                    table=(ref, r, c),
                )
            )
    return out


def seed_document(
    sessions: sessionmaker[Session],
    doc_id: str,
    fmt: DocumentFormat,
    spans: list[DocumentSpan],
    filename: str | None = None,
) -> None:
    """Register a parsed document with the given spans (bypasses ingestion/Docling)."""
    with sessions.begin() as s:
        s.add(
            DocumentRow(
                document_id=doc_id,
                filename=filename or f"{doc_id}.{fmt.value}",
                format=fmt.value,
                mime_type="application/x-test",
                sha256=doc_id.removeprefix("DOC-").ljust(64, "0"),
                size_bytes=1,
                version=1,
                supersedes=None,
                status="registered",
                error=None,
                storage_path="/dev/null",
                registered_at=datetime.now(UTC),
                parse_report=None,
            )
        )
    report = ParseReport(
        document_id=doc_id,
        filename=f"{doc_id}.{fmt.value}",
        format=fmt,
        sha256="0" * 64,
        status=DocumentStatus.PARSED,
        parser="test",
    )
    save_parse(sessions, ParsedDocument(report=report, spans=spans))


def llm_claim(**kw: Any) -> dict[str, Any]:
    c: dict[str, Any] = {
        "source_id": "S1",
        "subject": "Project Alpha",
        "predicate": "budget",
        "object_type": "money",
        "value": "5000000",
        "unit": None,
        "currency": "LKR",
        "modality": "approved",
        "valid_from": None,
        "valid_to": None,
        "time_expression": None,
        "negated": False,
        "evidence_quotes": ["approved allocation of LKR 5 million"],
        "confidence": 0.9,
    }
    c.update(kw)
    return c


def llm_entity(**kw: Any) -> dict[str, Any]:
    e: dict[str, Any] = {
        "source_id": "S1",
        "name": "Project Alpha",
        "entity_type": "project",
        "aliases": [],
    }
    e.update(kw)
    return e


def llm_json(
    claims: list[dict[str, Any]] | None = None, entities: list[dict[str, Any]] | None = None
) -> str:
    return json.dumps({"claims": claims or [], "entities": entities or []})
