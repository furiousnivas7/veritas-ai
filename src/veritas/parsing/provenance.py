"""Turn Docling's native provenance into VERITAS locations.

Behaviour verified against Docling 2.154 / docling-core 2.108 (see
tests/integration/test_parsing.py):

* PDF  - each item has ``prov[0].page_no`` (1-based physical page). Printed page
  labels are not exposed, so ``page_label`` stays None.
* DOCX - items have NO ``prov`` (Docling has no page model for DOCX). The stable
  locator is ``item.self_ref`` plus the heading path and table coordinates.
* XLSX - each sheet is a ``GroupItem`` (label ``sheet``, ``name`` = sheet name)
  and also a "page". An item's ``prov[0].bbox`` is in *cell units*: ``l``/``t``
  are the 0-based column/row of its top-left cell, ``r``/``b`` are exclusive.
  Table cell indices are relative to the table's origin.
"""

from __future__ import annotations

from docling_core.types.doc.document import DoclingDocument, GroupItem, NodeItem, TableItem
from docling_core.types.doc.labels import GroupLabel


def column_letter(index: int) -> str:
    """0 -> 'A', 25 -> 'Z', 26 -> 'AA'."""
    if index < 0:
        raise ValueError("column index must be >= 0")
    letters = ""
    index += 1
    while index:
        index, rem = divmod(index - 1, 26)
        letters = chr(ord("A") + rem) + letters
    return letters


def a1_range(col0: int, row0: int, col1: int, row1: int) -> str:
    """0-based inclusive corners -> 'C2' or 'C2:D5'."""
    start = f"{column_letter(col0)}{row0 + 1}"
    end = f"{column_letter(col1)}{row1 + 1}"
    return start if start == end else f"{start}:{end}"


def page_number(item: NodeItem) -> int | None:
    prov = getattr(item, "prov", None)
    return prov[0].page_no if prov else None


def sheet_name(doc: DoclingDocument, item: NodeItem) -> str | None:
    """Name of the XLSX sheet containing ``item``, or None."""
    node: NodeItem = item
    while node.parent is not None:
        parent = node.parent.resolve(doc)
        if isinstance(parent, GroupItem) and parent.label == GroupLabel.SHEET:
            return parent.name
        node = parent
    return None


def xlsx_item_range(item: NodeItem) -> str | None:
    """A1 range covered by an XLSX item, from its cell-unit bbox; None if absent."""
    prov = getattr(item, "prov", None)
    if not prov:
        return None
    b = prov[0].bbox
    left, top, right, bottom = int(b.l), int(b.t), int(b.r), int(b.b)
    if right <= left or bottom <= top:
        return None
    return a1_range(left, top, right - 1, bottom - 1)


def xlsx_cell_ref(
    table: TableItem, row: int, col: int, row_end: int | None = None, col_end: int | None = None
) -> str:
    """A1 address of a table cell in its sheet. ``row``/``col`` are relative to the
    table; ``row_end``/``col_end`` are exclusive ends for merged cells."""
    if not table.prov:
        raise ValueError(f"{table.self_ref} has no provenance; cannot locate it in the sheet")
    b = table.prov[0].bbox
    c0, r0 = int(b.l) + col, int(b.t) + row
    c1 = int(b.l) + (col_end - 1 if col_end else col)
    r1 = int(b.t) + (row_end - 1 if row_end else row)
    return a1_range(c0, r0, c1, r1)
