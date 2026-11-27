from __future__ import annotations

import pytest
from docling_core.types.doc.base import BoundingBox
from docling_core.types.doc.document import DoclingDocument, ProvenanceItem
from docling_core.types.doc.labels import DocItemLabel

from veritas.core.contracts import Document, Evidence
from veritas.parsing.provenance import a1_range, column_letter, xlsx_item_range
from veritas.parsing.service import ParseReport, _SpanBuilder


@pytest.mark.parametrize(
    ("i", "s"), [(0, "A"), (2, "C"), (25, "Z"), (26, "AA"), (701, "ZZ"), (702, "AAA")]
)
def test_column_letter(i: int, s: str) -> None:
    assert column_letter(i) == s


def test_column_letter_negative() -> None:
    with pytest.raises(ValueError):
        column_letter(-1)


def test_a1_range() -> None:
    assert a1_range(2, 1, 2, 1) == "C2"
    assert a1_range(0, 0, 3, 3) == "A1:D4"


def _doc(fmt: str) -> Document:
    return Document(
        document_id="DOC-0123456789ab",
        filename=f"x.{fmt}",
        format=fmt,
        mime_type="application/x",
        sha256="0" * 64,
        size_bytes=1,
        version=1,
    )


def _report(fmt: str) -> ParseReport:
    return ParseReport(
        document_id="DOC-0123456789ab",
        filename=f"x.{fmt}",
        format=fmt,
        sha256="0" * 64,
        status="failed",
        parser="test",
    )


def test_pdf_element_without_provenance_is_flagged_not_invented() -> None:
    """Missing source location: the span records why, and cannot become evidence."""
    dl = DoclingDocument(name="x")
    dl.add_text(label=DocItemLabel.TEXT, text="Budget is LKR 5 million.")  # no prov
    report = _report("pdf")
    (span,) = _SpanBuilder(_doc("pdf"), dl, report).build()

    assert span.location is None
    assert span.missing_provenance and "page" in span.missing_provenance
    assert report.spans_missing_provenance == 1
    with pytest.raises(ValueError, match="no location"):
        Evidence.from_span("EV-1", span)


def test_xlsx_text_item_located_from_bbox() -> None:
    dl = DoclingDocument(name="x")
    sheet = dl.add_group(label="sheet", name="Notes")
    prov = ProvenanceItem(page_no=1, bbox=BoundingBox(l=1, t=4, r=2, b=5), charspan=(0, 4))
    dl.add_text(label=DocItemLabel.TEXT, text="Note", prov=prov, parent=sheet)
    (span,) = _SpanBuilder(_doc("xlsx"), dl, _report("xlsx")).build()
    assert span.location is not None
    assert (span.location.sheet_name, span.location.cell_range) == ("Notes", "B5")


def test_xlsx_item_range_without_prov() -> None:
    dl = DoclingDocument(name="x")
    item = dl.add_text(label=DocItemLabel.TEXT, text="t")
    assert xlsx_item_range(item) is None
