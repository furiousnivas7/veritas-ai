"""Turn parsed spans into labelled source units for the model.

* A **text unit** is one located paragraph, list item, caption or footnote.
* A **row unit** is one data row of a table, shown as ``Header = value`` pairs,
  backed by that row's cell spans (header cells are context, not evidence).

Headings and titles are not sources; they appear as context for the units under
them. Spans without a location are excluded, because a claim built from them
could not be cited, and counted in ``SourcePlan.skipped_missing_provenance``.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal

from veritas.core.contracts import DocumentFormat, DocumentSpan, SpanKind
from veritas.extraction.grounding import NO_VALUES, YES_VALUES, date_spans

_TEXT_KINDS = {SpanKind.TEXT, SpanKind.LIST_ITEM, SpanKind.CAPTION, SpanKind.FOOTNOTE,
               SpanKind.OTHER}  # fmt: skip


Headers = tuple[DocumentSpan | None, ...]
_Planned = tuple[Literal["text", "table_row"], str, str, tuple[DocumentSpan, ...], Headers]


@dataclass(frozen=True)
class SourceUnit:
    source_id: str
    kind: Literal["text", "table_row"]
    text: str  # exactly what the model is shown as content
    context: str  # headings / sheet / columns: shown to the model, never grounds a claim
    spans: tuple[DocumentSpan, ...]  # evidence candidates
    labels: tuple[str | None, ...] = ()  # table rows: column header text of each span
    # Table rows: the located header cell of each span's column (None if the table
    # has no detected header row or the column has no header). Header cells are
    # supporting context for units/currency, never the claim's data-cell evidence.
    header_spans: tuple[DocumentSpan | None, ...] = ()

    def render(self) -> str:
        label = "row" if self.kind == "table_row" else "text"
        ctx = f"\ncontext: {self.context}" if self.context else ""
        return f"[{self.source_id}]{ctx}\n{label}: {self.text}"


@dataclass
class SourcePlan:
    batches: list[list[SourceUnit]] = field(default_factory=list)
    skipped_missing_provenance: int = 0
    units: int = 0
    tables_with_header: int = 0
    tables_without_header: int = 0
    tables_header_ambiguous: int = 0  # kept as data; see HeaderStatus
    header_decisions: list[dict[str, object]] = field(default_factory=list)


def _context(span: DocumentSpan, fmt: DocumentFormat) -> str:
    loc = span.location
    if loc is None:
        return ""
    parts = list(loc.heading_path)
    if fmt is DocumentFormat.XLSX and loc.sheet_name:
        parts.insert(0, f'sheet "{loc.sheet_name}"')
    return " > ".join(parts)


# --------------------------------------------------------------- table headers
#
# Policy: never drop a row unless it is positively identified as a header.
#
# Docling's TableCell.column_header cannot be trusted for this decision (checked on
# Docling 2.154): for DOCX and XLSX it marks row 0 as a header even in headerless
# tables, and for the sample PDF it marks no header at all. It is not used.
#
# Type contrast alone ("text above a number") is NOT evidence of a header: the data
# table [["Project Alpha", "pending"], ["Project Beta", "5"]] has exactly the same
# shape as [["Project", "Amount"], ["Project Alpha", "5"]]. A header therefore also
# needs every row-0 cell to look like a column label (see ``is_label_like``).
#
#   NO_HEADER  row 0 contains a value-like cell, or no column has a text cell over
#              value-like cells (all-text tables), or it is a single-row table
#   HEADER     the first row present is table row 0, all its cells are non-empty
#              label-like text, at least one column has text over values only, and
#              no row-0 cell repeats a value of its own column
#   AMBIGUOUS  anything else. Treated exactly like NO_HEADER: every row is data and
#              no header labels or header evidence are produced. Counted separately
#              so uncertain tables are visible in run statistics.


class HeaderStatus(StrEnum):
    HEADER = "header"
    NO_HEADER = "no_header"
    AMBIGUOUS = "ambiguous"


_VALUE_LIKE = re.compile(
    r"^\s*(?:[$€£₹]|LKR|USD|EUR|GBP|INR|Rs\.?)?\s*[-+(]?\d[\d,]*(?:\.\d+)?\)?\s*"
    r"(?:%|percent|k|m|mn|bn|million|billion|thousand|lakh|crore)?\s*$",
    re.IGNORECASE,
)

# Generic column-label vocabulary. A cell is label-like only if ALL its words are in
# this list (or connectors), or it names a unit in parentheses: "Amount (LKR)",
# "Staff (persons)". Proper names ("Project Alpha") and free text ("pending") are not.
HEADER_WORDS = frozenset(
    """
    name names title project projects programme program item items description details
    activity activities task tasks component components entity organisation organization
    institution department unit units division ministry agency owner owners responsible
    supervisor manager lead contact person staff team region district province location
    site sites country city address
    amount amounts total totals sum budget budgets cost costs price prices value values
    fee fees expenditure expenditures expense expenses spending spent allocation allocated
    funding funds fund grant grants revenue income balance payment payments
    proposed approved actual forecast forecasted estimated estimate planned revised
    variance difference change growth
    quantity qty count number no nos id code ref reference rate ratio percent percentage
    share score rank level size area length weight capacity duration hours days
    date dates year years month months quarter period fy start end deadline due from to
    completion completed status stage phase milestone milestones progress result results
    type category class kind modality source remarks remark notes note comments comment
    currency
    """.split()
)
_CONNECTORS = frozenset({"of", "the", "and", "or", "in", "per", "for", "by", "to", "at", "&", "/"})
# A trailing "(unit)" marks a column label only if the parentheses hold a currency
# or a measurement unit; "(Phase 2)" or "(2025)" do not.
_UNIT_IN_PARENS = re.compile(r"\(\s*([^()]{1,24}?)\s*\)\s*$")
UNIT_WORDS = frozenset(
    """
    lkr usd eur gbp inr rs rs. $ € £ ₹ mn million millions bn billion billions '000 000
    thousand thousands lakh lakhs crore crores persons person people staff no no. nos
    units unit sqm m2 sq.m km m kg tons tonnes t ha acres hours hrs days weeks months
    years % percent pct
    """.split()
)


def _unit_in_parens(text: str) -> str | None:
    m = _UNIT_IN_PARENS.search(text)
    if not m:
        return None
    inner = m.group(1).strip().lower()
    tokens = [tok for tok in re.split(r"[\s/,]+", inner) if tok]
    return inner if tokens and all(tok in UNIT_WORDS for tok in tokens) else None


def is_value_like(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if _VALUE_LIKE.match(t) or t.lower() in YES_VALUES | NO_VALUES:
        return True
    spans = date_spans(t)
    return bool(spans) and spans[0][0] == 0 and spans[-1][1] >= len(t.rstrip(". "))


def is_label_like(text: str) -> bool:
    """True for generic column labels ("Amount (LKR)", "Approved budget", "Date",
    "Beneficiaries (persons)"); False for data such as names ("Project Alpha",
    "Project Alpha (Phase 2)") or free text ("pending")."""
    t = text.strip()
    if not t or is_value_like(t) or len(t) > 60:
        return False
    has_unit = _unit_in_parens(t) is not None
    core = _UNIT_IN_PARENS.sub("", t).strip() if has_unit else t
    tokens = [w for w in re.findall(r"[A-Za-z]+|\d+|&|/", core.lower()) if w not in _CONNECTORS]
    # A year may qualify a label ("Budget 2025"); any other number reads as data
    # ("Phase 2", "Ward 5").
    if any(tok.isdigit() and not re.fullmatch(r"(19|20)\d{2}", tok) for tok in tokens):
        return False
    words = [tok for tok in tokens if not tok.isdigit()]
    if len(words) > 6 or (not words and not has_unit):
        return False
    return has_unit or all(w in HEADER_WORDS for w in words)


HeaderPolicy = Literal["detect", "never"]


def classify_header_row(
    by_row: dict[int, dict[int, DocumentSpan]],
) -> tuple[HeaderStatus, str]:
    """(status, reason). See the policy table above."""
    rows = sorted(by_row)
    if len(rows) < 2:
        return HeaderStatus.NO_HEADER, "single-row table"
    first = by_row[rows[0]]
    below = [by_row[r] for r in rows[1:]]
    if not first or any(is_value_like(c.text) for c in first.values()):
        return HeaderStatus.NO_HEADER, "first row contains a value-like cell"

    def contrast(col: int) -> bool:
        """Text in row 0 over a column whose every present cell is value-like."""
        cells = [r[col].text for r in below if col in r and r[col].text.strip()]
        return bool(cells) and all(is_value_like(t) for t in cells)

    if not any(contrast(col) for col in first):
        return HeaderStatus.NO_HEADER, "no column has text over values"
    if rows[0] != 0:
        return HeaderStatus.AMBIGUOUS, "table row 0 is missing"
    unlabeled = [c.text for c in first.values() if not is_label_like(c.text)]
    if unlabeled:
        return HeaderStatus.AMBIGUOUS, f"first-row cells are not column labels: {unlabeled[:3]}"
    for col, cell in first.items():
        label = cell.text.strip().casefold()
        if any(col in r and r[col].text.strip().casefold() == label for r in below):
            return HeaderStatus.AMBIGUOUS, f"first-row cell {cell.text!r} repeats a column value"
    # Content evidence only: generic labels over a value column. This cannot rule out
    # a data row that happens to consist of label words (see docs/data_model.md);
    # the decision is recorded in the plan so it can be audited or overridden.
    return HeaderStatus.HEADER, "first-row cells are all column labels over a value column"


def detect_header_row(by_row: dict[int, dict[int, DocumentSpan]]) -> HeaderStatus:
    return classify_header_row(by_row)[0]


def _row_units(
    cells: list[DocumentSpan], fmt: DocumentFormat, policy: HeaderPolicy = "detect"
) -> tuple[HeaderStatus, str, list[tuple[str, str, tuple[DocumentSpan, ...], Headers]]]:
    by_row: dict[int, dict[int, DocumentSpan]] = defaultdict(dict)
    for c in cells:
        assert c.location is not None and c.location.table_row is not None
        by_row[c.location.table_row][c.location.table_col or 0] = c
    rows = sorted(by_row)
    if not rows:
        return HeaderStatus.NO_HEADER, "empty table", []
    if policy == "never":
        status, reason = HeaderStatus.NO_HEADER, "header detection disabled (policy 'never')"
    else:
        status, reason = classify_header_row(by_row)
    has_header = status is HeaderStatus.HEADER  # AMBIGUOUS keeps every row as data
    header = by_row[rows[0]] if has_header else {}
    data_rows = rows[1:] if has_header else rows
    out = []
    for r in data_rows:
        cols = by_row[r]
        pairs = []
        for col in sorted(cols):
            h = header.get(col)
            pairs.append(f"{h.text} = {cols[col].text}" if h else cols[col].text)
        first = cols[min(cols)]
        ctx = _context(first, fmt)
        headers = " | ".join(header[c].text for c in sorted(header))
        if headers:
            ctx = f"{ctx} > columns: {headers}" if ctx else f"columns: {headers}"
        hdr_spans = tuple(header.get(c) for c in sorted(cols))
        out.append((" | ".join(pairs), ctx, tuple(cols[c] for c in sorted(cols)), hdr_spans))
    return status, reason, out


def plan_sources(
    spans: list[DocumentSpan],
    fmt: DocumentFormat,
    batch_chars: int = 1800,
    header_policy: HeaderPolicy = "detect",
) -> SourcePlan:
    """``header_policy="never"`` treats every table row as data (guaranteed
    preservation of first rows, at the cost of all header labels)."""
    plan = SourcePlan()
    raw: list[tuple[Literal["text", "table_row"], str, str, tuple[DocumentSpan, ...]]] = []
    tables: dict[str, list[DocumentSpan]] = defaultdict(list)
    table_first_order: dict[str, int] = {}

    for s in sorted(spans, key=lambda s: s.order):
        if s.location is None:
            if s.kind in _TEXT_KINDS or s.kind is SpanKind.TABLE_CELL:
                plan.skipped_missing_provenance += 1
            continue
        if s.kind in _TEXT_KINDS and s.text.strip():
            raw.append(("text", s.text, _context(s, fmt), (s,)))
        elif s.kind is SpanKind.TABLE_CELL and s.location.table_ref:
            ref = s.location.table_ref
            if ref not in table_first_order:
                table_first_order[ref] = len(raw)
                raw.append(("table_row", "", ref, ()))  # placeholder keeps reading order
            tables[ref].append(s)

    ordered: list[_Planned] = []
    for kind, text, ctx, sp in raw:
        if kind == "table_row" and not sp:
            status, reason, rows = _row_units(tables[ctx], fmt, header_policy)
            first_row = min((c.location.table_row or 0) for c in tables[ctx] if c.location)
            plan.header_decisions.append(
                {
                    "table_ref": ctx,
                    "status": status.value,
                    "reason": reason,
                    # The row-0 cells withheld from the data rows (HEADER only). They
                    # remain in document_spans and are carried as header_spans.
                    "header_span_ids": sorted(
                        c.span_id
                        for c in tables[ctx]
                        if c.location and c.location.table_row == first_row
                    )
                    if status is HeaderStatus.HEADER
                    else [],
                }
            )
            if status is HeaderStatus.HEADER:
                plan.tables_with_header += 1
            elif status is HeaderStatus.AMBIGUOUS:
                plan.tables_header_ambiguous += 1
            else:
                plan.tables_without_header += 1
            ordered.extend(("table_row", t, c, cs, hs) for t, c, cs, hs in rows)
        else:
            ordered.append((kind, text, ctx, sp, ()))

    batch: list[SourceUnit] = []
    size = 0
    for kind, text, ctx, sp, hdrs in ordered:
        length = len(text) + len(ctx)
        if batch and size + length > batch_chars:
            plan.batches.append(batch)
            batch, size = [], 0
        labels = tuple(h.text if h else None for h in hdrs)
        unit = SourceUnit(f"S{len(batch) + 1}", kind, text, ctx, sp, labels, hdrs)
        batch.append(unit)
        size += length
    if batch:
        plan.batches.append(batch)
    plan.units = sum(len(b) for b in plan.batches)
    return plan
