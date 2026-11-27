"""Persist parse results: spans into ``document_spans``, report onto ``documents``."""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, sessionmaker

from veritas.core.contracts import DocumentSpan, SourceLocation, SpanKind
from veritas.core.db import DocumentRow, DocumentSpanRow
from veritas.parsing.service import ParsedDocument


def save_parse(session_factory: sessionmaker[Session], parsed: ParsedDocument) -> None:
    """Replace any earlier spans for the document (re-parsing is idempotent)."""
    report = parsed.report
    with session_factory.begin() as s:
        row = s.get(DocumentRow, report.document_id)
        if row is None:
            raise KeyError(f"document {report.document_id} is not registered")
        s.execute(delete(DocumentSpanRow).where(DocumentSpanRow.document_id == report.document_id))
        for span in parsed.spans:
            loc = span.location
            s.add(
                DocumentSpanRow(
                    span_id=span.span_id,
                    document_id=span.document_id,
                    document_version=span.document_version,
                    kind=span.kind.value,
                    order=span.order,
                    text=span.text,
                    parent_span_id=span.parent_span_id,
                    page_number=loc.page_number if loc else None,
                    sheet_name=loc.sheet_name if loc else None,
                    cell_range=loc.cell_range if loc else None,
                    element_id=loc.element_id if loc else None,
                    location_json=loc.model_dump(mode="json") if loc else None,
                    missing_provenance=span.missing_provenance,
                )
            )
        row.status = report.status.value
        row.error = "; ".join(report.errors) or None
        row.parse_report = report.model_dump(mode="json")


def load_spans(session_factory: sessionmaker[Session], document_id: str) -> list[DocumentSpan]:
    with session_factory() as s:
        rows = s.scalars(
            select(DocumentSpanRow)
            .where(DocumentSpanRow.document_id == document_id)
            .order_by(DocumentSpanRow.order)
        ).all()
        return [
            DocumentSpan(
                span_id=r.span_id,
                document_id=r.document_id,
                document_version=r.document_version,
                kind=SpanKind(r.kind),
                text=r.text,
                order=r.order,
                parent_span_id=r.parent_span_id,
                location=SourceLocation.model_validate(r.location_json)
                if r.location_json
                else None,
                missing_provenance=r.missing_provenance,
            )
            for r in rows
        ]
