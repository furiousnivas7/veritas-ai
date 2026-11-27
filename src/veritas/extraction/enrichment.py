"""Deterministic, evidence-grounded enrichment of claim currency and period.

No model call. Everything here reads only text the validator hands in, which is
always inside the claim's approved evidence scope: the quoted span, the quote's own
sentence, or the cited table row and the value column's own header. Headings,
sheet names, other sentences/rows/columns and document metadata are never passed
in, so they can never supply a value.

Precision is preserved: a year stays "2025", a month "2025-03", a day
"2025-03-15". Nothing is padded or invented. Ambiguous inputs produce no value
and a recorded reason; they never produce a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from veritas.extraction import grounding as g

# ------------------------------------------------------------------ periods

_MONTHS = {
    name: i + 1
    for i, name in enumerate(
        ["january", "february", "march", "april", "may", "june", "july", "august",
         "september", "october", "november", "december"]
    )
}  # fmt: skip
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})
_MONTHS["sept"] = 9
_MON = r"(?:" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?"
_YEAR = r"[12]\d{3}"
_DAY = r"[0-3]?\d(?:st|nd|rd|th)?"
_DASH = r"(?:-|–|—|to|until|through|till)"

# A single date expression (any precision). Order matters: longest forms first.
_DATE_EXPR = (
    rf"(?:{_DAY}\s+(?:of\s+)?{_MON},?\s+{_YEAR}"  # 15 January 2025
    rf"|{_MON}\s+{_DAY},?\s+{_YEAR}"  # January 15, 2025
    rf"|{_MON},?\s+{_YEAR}"  # January 2025
    rf"|{_YEAR}[-/.]\d{{1,2}}[-/.]\d{{1,2}}"  # 2025-01-15
    rf"|\d{{1,2}}[-/.]\d{{1,2}}[-/.]{_YEAR}"  # 15/01/2025
    rf"|{_YEAR}[-/]\d{{1,2}}(?![\d/-])"  # 2025-01
    rf"|\d{{1,2}}/{_YEAR}"  # 01/2025
    rf"|{_YEAR})"  # 2025
)
_B = r"(?<![\w./-])"  # token start
_E = r"(?![\w/]|[.,]\d)"  # token end

_RANGE_MONTHS = re.compile(
    rf"{_B}(?:from\s+)?(?P<m1>{_MON})(?:\s+(?P<y1>{_YEAR}))?\s*{_DASH}\s*(?P<m2>{_MON})\s+(?P<y2>{_YEAR}){_E}",
    re.I,
)
_RANGE_YEARS = re.compile(
    rf"{_B}(?:(?:from\s+)?(?P<a>{_YEAR})\s*{_DASH}\s*(?P<b>{_YEAR})"
    rf"|between\s+(?P<c>{_YEAR})\s+and\s+(?P<d>{_YEAR})){_E}",
    re.I,
)
_FISCAL = re.compile(rf"{_B}FY\s*'?(?P<y>{_YEAR})(?:\s*[/-]\s*(?P<y2>\d{{2}}|{_YEAR}))?{_E}", re.I)
_CUED = re.compile(
    rf"{_B}(?P<cue>in|for|during|throughout|as\s+of|as\s+at|on)\s+(?P<expr>{_DATE_EXPR}){_E}",
    re.I,
)
_UNSUPPORTED = re.compile(
    rf"{_B}(?:(?:Q[1-4]|H[12])\s*{_YEAR}|(?:since|until|by|before|after|from)\s+(?:{_MON}\s+)?{_YEAR}){_E}",
    re.I,
)


@dataclass(frozen=True)
class Period:
    valid_from: str
    valid_to: str
    start: int  # offsets of the expression in the scanned text
    end: int
    expression: str
    precision_reduced: bool = False

    @property
    def key(self) -> tuple[str, str]:
        return self.valid_from, self.valid_to

    @property
    def label(self) -> str:
        return f"{self.valid_from}..{self.valid_to}"


@dataclass
class PeriodScan:
    periods: list[Period] = field(default_factory=list)
    invalid: list[str] = field(default_factory=list)  # impossible dates seen
    unsupported: list[str] = field(default_factory=list)  # "Q1 2025", "since 2024"


INVALID = "invalid"


def _real(y: int, m: int | None = None, d: int | None = None) -> bool:
    try:
        date(y, 1 if m is None else m, 1 if d is None else d)
    except ValueError:
        return False
    return True


def _fmt(y: int, m: int | None = None, d: int | None = None) -> str:
    return f"{y:04d}" + (f"-{m:02d}" if m else "") + (f"-{d:02d}" if d else "")


def parse_date_expression(expr: str) -> tuple[str, str, bool] | str | None:
    """A standalone date expression -> (valid_from, valid_to, precision_reduced), the
    string ``INVALID`` for impossible dates, or None if it is not a date at all.

    Numeric day/month order that cannot be decided (01/02/2025) is reduced to the
    year, which both readings share; nothing is guessed."""
    t = expr.strip().rstrip(".,")
    low = t.lower()
    m = re.fullmatch(rf"({_DAY})\s+(?:of\s+)?({_MON}),?\s+({_YEAR})", low)
    if m:
        d, mo, y = (
            int(re.sub(r"\D", "", m.group(1))),
            _MONTHS[m.group(2).rstrip(".")],
            int(m.group(3)),
        )
        return (_fmt(y, mo, d), _fmt(y, mo, d), False) if _real(y, mo, d) else INVALID
    m = re.fullmatch(rf"({_MON})\s+({_DAY}),?\s+({_YEAR})", low)
    if m:
        mo, d, y = (
            _MONTHS[m.group(1).rstrip(".")],
            int(re.sub(r"\D", "", m.group(2))),
            int(m.group(3)),
        )
        return (_fmt(y, mo, d), _fmt(y, mo, d), False) if _real(y, mo, d) else INVALID
    m = re.fullmatch(rf"({_MON}),?\s+({_YEAR})", low)
    if m:
        mo, y = _MONTHS[m.group(1).rstrip(".")], int(m.group(2))
        return _fmt(y, mo), _fmt(y, mo), False
    m = re.fullmatch(rf"({_YEAR})([-/.])(\d{{1,2}})\2(\d{{1,2}})", t)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(3)), int(m.group(4))
        return (_fmt(y, mo, d), _fmt(y, mo, d), False) if _real(y, mo, d) else INVALID
    m = re.fullmatch(rf"(\d{{1,2}})([-/.])(\d{{1,2}})\2({_YEAR})", t)
    if m:
        a, b, y = int(m.group(1)), int(m.group(3)), int(m.group(4))
        readings = {(y, mo, d) for mo, d in ((b, a), (a, b)) if _real(y, mo, d)}
        if not readings:
            return INVALID
        if len(readings) == 1:
            (yy, mo, d) = readings.pop()
            return _fmt(yy, mo, d), _fmt(yy, mo, d), False
        return _fmt(y), _fmt(y), True
    m = re.fullmatch(rf"({_YEAR})[-/](\d{{1,2}})", t)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        return (_fmt(y, mo), _fmt(y, mo), False) if _real(y, mo) else INVALID
    m = re.fullmatch(rf"(\d{{1,2}})/({_YEAR})", t)
    if m:
        mo, y = int(m.group(1)), int(m.group(2))
        return (_fmt(y, mo), _fmt(y, mo), False) if _real(y, mo) else INVALID
    if re.fullmatch(_YEAR, t):
        return t, t, False
    return None


def find_periods(text: str) -> PeriodScan:
    """Explicit periods stated in prose. A bare year counts only after a cue word
    ("in", "for", "during", "as of", "on") or inside a range / fiscal-year form, so
    "42 staff" or "2025 participants" never becomes a period."""
    scan = PeriodScan()
    taken: list[tuple[int, int]] = []

    def free(a: int, b: int) -> bool:
        return all(b <= x or a >= y for x, y in taken)

    for m in _RANGE_MONTHS.finditer(text):
        y2 = int(m["y2"])
        y1 = int(m["y1"]) if m["y1"] else y2
        m1, m2 = _MONTHS[m["m1"].lower().rstrip(".")], _MONTHS[m["m2"].lower().rstrip(".")]
        if (y1, m1) <= (y2, m2) and free(*m.span()):
            scan.periods.append(Period(_fmt(y1, m1), _fmt(y2, m2), *m.span(), m.group(0)))
            taken.append(m.span())
    for m in _RANGE_YEARS.finditer(text):
        a, b = (m["a"], m["b"]) if m["a"] else (m["c"], m["d"])
        if int(a) <= int(b) and free(*m.span()):
            scan.periods.append(Period(a, b, *m.span(), m.group(0)))
            taken.append(m.span())
    for m in _FISCAL.finditer(text):
        if not free(*m.span()):
            continue
        y = int(m["y"])
        if m["y2"]:
            y2 = int(m["y2"]) if len(m["y2"]) == 4 else (y // 100) * 100 + int(m["y2"])
            if y2 != y + 1:
                continue  # "FY 2024/27" is not a fiscal year
            scan.periods.append(Period(_fmt(y), _fmt(y2), *m.span(), m.group(0)))
        else:
            scan.periods.append(Period(_fmt(y), _fmt(y), *m.span(), m.group(0)))
        taken.append(m.span())
    for m in _CUED.finditer(text):
        a, b = m.span("expr")
        if not free(a, b):
            continue
        parsed = parse_date_expression(m["expr"])
        if parsed == INVALID:
            scan.invalid.append(m["expr"])
        elif isinstance(parsed, tuple):
            vf, vt, reduced = parsed
            # The expression includes its cue word: "for 2025", "in March 2025".
            scan.periods.append(Period(vf, vt, m.start(), b, m.group(0), reduced))
        taken.append((a, b))
    for m in _UNSUPPORTED.finditer(text):
        if free(*m.span()):
            scan.unsupported.append(m.group(0))
    scan.periods.sort(key=lambda p: p.start)
    return scan


# Column headers that mark a row's date cell as the row's period.
TIME_HEADER_WORDS = frozenset({"date", "year", "period", "month", "fy", "as of", "as at"})
START_HEADER_WORDS = frozenset({"start", "from", "begin", "beginning", "commencement"})
END_HEADER_WORDS = frozenset({"end", "until", "close", "closing", "completion"})


def header_time_role(label: str) -> str | None:
    """'period' / 'start' / 'end' if a column header names a time concept, else None."""
    low = label.casefold()
    words = set(re.findall(r"[a-z]+", low))
    if words & START_HEADER_WORDS:
        return "start"
    if words & END_HEADER_WORDS:
        return "end"
    if words & TIME_HEADER_WORDS or "as of" in low or "as at" in low:
        return "period"
    return None


# ----------------------------------------------------------------- currency


@dataclass
class CurrencyFinding:
    codes: set[str] = field(default_factory=set)  # resolved, attached / header currencies
    ambiguous: set[str] = field(default_factory=set)  # unresolved forms: "rs", "$"
    support: dict[str, list[tuple[int, int, int]]] = field(default_factory=dict)
    # code -> [(locator index, start, end)] where the code was found


def attached_currencies(
    text: str, start: int, end: int, defaults: g.CurrencyDefaults | None
) -> list[g.CurrencyMention]:
    return g.currencies_attached(text, start, end, defaults)


def value_occurrences(value: Decimal, text: str, lo: int, hi: int) -> list[tuple[int, int]]:
    """Spans of quantities equal to ``value`` within text[lo:hi]."""
    target = g.canonical(value)
    return [(a, b) for a, b, v in g.number_spans(text) if v == target and a >= lo and b <= hi]
