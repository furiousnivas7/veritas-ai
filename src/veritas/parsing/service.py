"""Document parsing with Docling -> provenance-carrying DocumentSpans.

Output per document:
* ``DocumentSpan`` records (headings, paragraphs, list items, tables, table cells)
* a ``ParseReport`` (the inspection report)
* normalised JSON at ``<parsed_dir>/<document_id>.json`` and the raw Docling
  export at ``<parsed_dir>/<document_id>.docling.json`` (for re-processing)

A location is only recorded when Docling exposes it. If it does not, the span
gets ``missing_provenance`` with the reason; nothing is inferred or invented.
"""

from __future__ import annotations

import importlib.metadata
import json
import logging
import time
from datetime import datetime
from pathlib import Path

from docling.datamodel.base_models import ConversionStatus, InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc.document import (
    DoclingDocument,
    ListItem,
    NodeItem,
    PictureItem,
    SectionHeaderItem,
    TableItem,
    TextItem,
    TitleItem,
)
from docling_core.types.doc.labels import DocItemLabel, GroupLabel
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from veritas.core.contracts import (
    Document,
    DocumentFormat,
    DocumentSpan,
    DocumentStatus,
    SourceLocation,
    SpanKind,
    utcnow,
)
from veritas.parsing import provenance as prov

log = logging.getLogger(__name__)

_LABEL_KIND = {
    DocItemLabel.TITLE: SpanKind.TITLE,
    DocItemLabel.SECTION_HEADER: SpanKind.HEADING,
    DocItemLabel.TEXT: SpanKind.TEXT,
    DocItemLabel.PARAGRAPH: SpanKind.TEXT,
    DocItemLabel.LIST_ITEM: SpanKind.LIST_ITEM,
    DocItemLabel.CAPTION: SpanKind.CAPTION,
    DocItemLabel.FOOTNOTE: SpanKind.FOOTNOTE,
}


class ParseReport(BaseModel):
    """Inspection report for one parsed document."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    filename: str
    format: DocumentFormat
    sha256: str
    status: DocumentStatus
    total_pages: int | None = Field(default=None, description="PDF only")
    total_sheets: int | None = Field(default=None, description="XLSX only")
    elements_extracted: int = 0
    tables_extracted: int = 0
    table_cells_extracted: int = 0
    headings_extracted: int = 0
    pictures_skipped: int = 0
    spans_missing_provenance: int = 0
    provenance_available: bool = False
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    duration_seconds: float = 0.0
    parser: str
    parsed_at: datetime = Field(default_factory=utcnow)


class ParsedDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    report: ParseReport
    spans: list[DocumentSpan]
    # Raw Docling output, kept in memory for export; not part of the JSON contract.
    _docling: DoclingDocument | None = PrivateAttr(default=None)


def _parser_version() -> str:
    return f"docling {importlib.metadata.version('docling')}"


class DocumentParser:
    """Wraps a single, lazily created Docling converter (model loading is slow)."""

    def __init__(self, pdf_ocr: bool = False) -> None:
        self._pdf_ocr = pdf_ocr
        self._converter: DocumentConverter | None = None

    @property
    def converter(self) -> DocumentConverter:
        if self._converter is None:
            opts = PdfPipelineOptions(do_ocr=self._pdf_ocr, do_table_structure=True)
            self._converter = DocumentConverter(
                allowed_formats=[InputFormat.PDF, InputFormat.DOCX, InputFormat.XLSX],
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)},
            )
        return self._converter

    def parse(self, document: Document, path: Path) -> ParsedDocument:
        started = time.perf_counter()
        report = ParseReport(
            document_id=document.document_id,
            filename=document.filename,
            format=document.format,
            sha256=document.sha256,
            status=DocumentStatus.FAILED,
            parser=_parser_version(),
        )
        try:
            result = self.converter.convert(path, raises_on_error=False)
        except Exception as e:  # Docling raises various types for unreadable input
            report.errors.append(f"{type(e).__name__}: {e}")
            report.duration_seconds = round(time.perf_counter() - started, 3)
            log.warning(
                "parse failed", extra={"document_id": document.document_id, "error": str(e)}
            )
            return ParsedDocument(report=report, spans=[])

        report.errors.extend(str(getattr(e, "error_message", e)) for e in result.errors)
        if result.status not in (ConversionStatus.SUCCESS, ConversionStatus.PARTIAL_SUCCESS):
            report.errors.append(f"docling status: {result.status.value}")
            report.duration_seconds = round(time.perf_counter() - started, 3)
            log.warning("parse failed", extra={"document_id": document.document_id})
            return ParsedDocument(report=report, spans=[])
        if result.status is ConversionStatus.PARTIAL_SUCCESS:
            report.warnings.append("docling reported partial success")

        spans = _SpanBuilder(document, result.document, report).build()
        report.status = DocumentStatus.PARSED
        report.provenance_available = bool(spans) and report.spans_missing_provenance == 0
        if not spans:
            report.warnings.append("no text elements extracted (empty or image-only document)")
        report.duration_seconds = round(time.perf_counter() - started, 3)
        log.info(
            "document parsed",
            extra={
                "document_id": document.document_id,
                "elements": report.elements_extracted,
                "tables": report.tables_extracted,
                "missing_provenance": report.spans_missing_provenance,
                "seconds": report.duration_seconds,
            },
        )
        parsed = ParsedDocument(report=report, spans=spans)
        parsed._docling = result.document
        return parsed

    def export(self, parsed: ParsedDocument, out_dir: Path) -> Path:
        """Write normalised JSON (and the raw Docling JSON if available)."""
        out_dir.mkdir(parents=True, exist_ok=True)
        doc_id = parsed.report.document_id
        out = out_dir / f"{doc_id}.json"
        out.write_text(parsed.model_dump_json(indent=2))
        if parsed._docling is not None:
            raw = out_dir / f"{doc_id}.docling.json"
            raw.write_text(json.dumps(parsed._docling.export_to_dict()))
        return out


# --------------------------------------------------------------------------- spans


class _SpanBuilder:
    """Walks a DoclingDocument in reading order and emits DocumentSpans."""

    def __init__(self, document: Document, doc: DoclingDocument, report: ParseReport) -> None:
        self.document = document
        self.doc = doc
        self.fmt = document.format
        self.report = report
        self.spans: list[DocumentSpan] = []
        self.headings: list[tuple[int, str]] = []  # (level, text) stack

    def build(self) -> list[DocumentSpan]:
        if self.fmt is DocumentFormat.PDF:
            self.report.total_pages = len(self.doc.pages)
        elif self.fmt is DocumentFormat.XLSX:
            self.report.total_sheets = sum(
                1 for g in self.doc.groups if g.label == GroupLabel.SHEET
            )

        for item, _level in self.doc.iterate_items():
            if isinstance(item, PictureItem):
                self.report.pictures_skipped += 1
            elif isinstance(item, TableItem):
                self._table(item)
            elif isinstance(item, TextItem) and item.text.strip():
                self._text(item)
        return self.spans

    # -- element handlers

    def _text(self, item: TextItem) -> None:
        if isinstance(item, ListItem):
            kind = SpanKind.LIST_ITEM
        else:
            kind = _LABEL_KIND.get(item.label, SpanKind.OTHER)

        # A heading's path is the headings above it; update the stack afterwards.
        loc, missing = self._locate(item, self._heading_path())
        if isinstance(item, TitleItem):
            self.headings = [(0, item.text)]
        elif isinstance(item, SectionHeaderItem):
            self.headings = [h for h in self.headings if h[0] < item.level]
            self.headings.append((item.level, item.text))
        if kind in (SpanKind.TITLE, SpanKind.HEADING):
            self.report.headings_extracted += 1

        self._add(kind, item.text, item.self_ref, loc, missing)
        self.report.elements_extracted += 1

    def _table(self, table: TableItem) -> None:
        path = self._heading_path()
        loc, missing = self._locate(table, path)
        table_span = self._add(
            SpanKind.TABLE, table.export_to_markdown(self.doc), table.self_ref, loc, missing
        )
        self.report.tables_extracted += 1
        self.report.elements_extracted += 1

        for cell in table.data.table_cells:
            text = cell.text.strip()
            if not text:
                continue
            r, c = cell.start_row_offset_idx, cell.start_col_offset_idx
            cell_range = None
            if self.fmt is DocumentFormat.XLSX and table.prov:
                cell_range = prov.xlsx_cell_ref(
                    table, r, c, cell.end_row_offset_idx, cell.end_col_offset_idx
                )
            cell_loc, cell_missing = self._locate(
                table, path, table_ref=table.self_ref, row=r, col=c, cell_range=cell_range
            )
            self._add(
                SpanKind.TABLE_CELL,
                text,
                table.self_ref,
                cell_loc,
                cell_missing,
                suffix=f":r{r}c{c}",
                parent=table_span,
            )
            self.report.table_cells_extracted += 1

    # -- helpers

    def _heading_path(self) -> list[str]:
        return [text for _, text in self.headings]

    def _locate(
        self,
        item: NodeItem,
        heading_path: list[str],
        *,
        table_ref: str | None = None,
        row: int | None = None,
        col: int | None = None,
        cell_range: str | None = None,
    ) -> tuple[SourceLocation | None, str | None]:
        """Location from Docling provenance, or (None, reason) if it is not exposed."""
        common = {
            "format": self.fmt,
            "element_id": item.self_ref,
            "heading_path": heading_path,
            "table_ref": table_ref,
            "table_row": row,
            "table_col": col,
        }
        if self.fmt is DocumentFormat.PDF:
            page = prov.page_number(item)
            if page is None:
                return None, "docling exposed no page provenance for this PDF element"
            return SourceLocation.model_validate({**common, "page_number": page}), None

        if self.fmt is DocumentFormat.XLSX:
            sheet = prov.sheet_name(self.doc, item)
            cells = cell_range or prov.xlsx_item_range(item)
            if not sheet or not cells:
                return None, "docling exposed no sheet/cell provenance for this XLSX element"
            return SourceLocation.model_validate(
                {**common, "sheet_name": sheet, "cell_range": cells}
            ), None

        # DOCX: structural location only; Docling has no page model for DOCX.
        return SourceLocation.model_validate(common), None

    def _add(
        self,
        kind: SpanKind,
        text: str,
        element_id: str,
        location: SourceLocation | None,
        missing: str | None,
        *,
        suffix: str = "",
        parent: str | None = None,
    ) -> str:
        span_id = f"{self.document.document_id}:{element_id}{suffix}"
        self.spans.append(
            DocumentSpan(
                span_id=span_id,
                document_id=self.document.document_id,
                document_version=self.document.version,
                kind=kind,
                text=text,
                order=len(self.spans),
                parent_span_id=parent,
                location=location,
                missing_provenance=missing,
            )
        )
        if location is None:
            self.report.spans_missing_provenance += 1
        return span_id
