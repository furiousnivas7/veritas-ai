"""Deterministic checks of model output against source text.

Nothing here calls a model. Each function answers one question ("is this quote in
the source?", "does this number appear in the quote?") so that validation is
reproducible and independent of which LLM produced the candidate.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Literal

from veritas.core.contracts import Modality, parse_partial_date

# --------------------------------------------------------------------------- quotes


def locate(quote: str, text: str) -> tuple[int, int] | None:
    """Character span of ``quote`` in ``text``.

    Exact match first, then case-insensitive with flexible whitespace (PDF text
    often has different line breaks). Offsets always index the *original* text,
    so the stored evidence is the source's own characters, never the model's.
    """
    q = quote.strip()
    if not q:
        return None
    i = text.find(q)
    if i >= 0:
        return i, i + len(q)
    pattern = r"\s+".join(re.escape(tok) for tok in q.split())
    m = re.search(pattern, text, flags=re.IGNORECASE)
    return (m.start(), m.end()) if m else None


def contains(needle: str, haystack: str) -> bool:
    return locate(needle, haystack) is not None


# -------------------------------------------------------------------------- numbers

_SCALES = {
    "thousand": Decimal(10**3), "k": Decimal(10**3),
    "lakh": Decimal(10**5), "lakhs": Decimal(10**5),
    "million": Decimal(10**6), "millions": Decimal(10**6), "mn": Decimal(10**6),
    "m": Decimal(10**6), "mil": Decimal(10**6),
    "crore": Decimal(10**7), "crores": Decimal(10**7),
    "billion": Decimal(10**9), "billions": Decimal(10**9), "bn": Decimal(10**9),
    "b": Decimal(10**9),
}  # fmt: skip
_NUMBER = re.compile(
    r"(?<![\w.])(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?:\s*("
    + "|".join(sorted(_SCALES, key=len, reverse=True))
    + r")\b)?",
    re.IGNORECASE,
)


def canonical(d: Decimal) -> Decimal:
    """Normalise without exponent notation for whole numbers: 5000000, not 5E+6."""
    d = d.normalize()
    return d.quantize(Decimal(1)) if d == d.to_integral_value() else d


def parse_numbers(text: str) -> list[Decimal]:
    """All quantities in ``text`` with scale words applied.

    'LKR 5 million' -> [5000000]; '5,500,000' -> [5500000]; 'Rs. 2.5bn' -> [2500000000].
    Date expressions are removed first, so the parts of '2025-01-15', '15/01/2025' or
    '15 January 2025' are never read as quantities ("15 projects completed on
    2025-01-15" -> [15]). A bare year ("in 2025") is still a number: it cannot be
    told apart from a count without more context.
    """
    return [value for _, _, value in number_spans(text)]


def number_spans(text: str) -> list[tuple[int, int, Decimal]]:
    """(start, end, value) of every quantity, as in ``parse_numbers`` (dates masked).
    The span covers the digits and any scale word ("7 million")."""
    masked = mask_dates(text)
    out: list[tuple[int, int, Decimal]] = []
    for m in _NUMBER.finditer(masked):
        n = Decimal(m.group(1).replace(",", "") + (m.group(2) or ""))
        if m.group(3):
            n *= _SCALES[m.group(3).lower()]
        out.append((m.start(), m.end(), canonical(n)))
    return out


def parse_value(value: str) -> Decimal | None:
    """Parse a model-supplied numeric value; tolerates separators, codes and scale words."""
    cleaned = re.sub(r"[A-Za-z$€£₹]{0,3}\.?\s*(?=\d)", "", value.strip(), count=1)
    try:
        return canonical(Decimal(cleaned.replace(",", "")))
    except InvalidOperation:
        nums = parse_numbers(value)
        return nums[0] if len(nums) == 1 else None


def number_in(value: Decimal, text: str) -> bool:
    target = canonical(value)
    return any(n == target for n in parse_numbers(text))


# ------------------------------------------------------------------------ currency
#
# Recognised currency forms. ISO codes must appear in upper case in the source (so the
# words "cad" or "aud" are never currencies). Symbols and words that several
# currencies share ($, Rs., rupees, dollars, pounds, ¥) resolve to NOTHING unless the
# run is explicitly configured with a default for them (``currency_symbol_defaults``,
# e.g. {"Rs": "LKR"}); the effective defaults are recorded in each run's options.

CURRENCY_CODES = frozenset(
    {"LKR", "USD", "EUR", "GBP", "INR", "JPY", "AUD", "CAD", "SGD", "CNY", "PKR", "NPR",
     "BDT", "MVR", "CHF", "AED", "SAR", "MYR", "THB", "NZD", "HKD", "KRW"}
)  # fmt: skip
UNAMBIGUOUS_CURRENCY_FORMS: dict[str, str] = {
    "sri lankan rupees": "LKR", "sri lankan rupee": "LKR",
    "us$": "USD", "us dollars": "USD", "us dollar": "USD", "u.s. dollars": "USD",
    "a$": "AUD", "au$": "AUD", "c$": "CAD", "s$": "SGD", "hk$": "HKD", "nz$": "NZD",
    "€": "EUR", "euro": "EUR", "euros": "EUR",
    "£": "GBP", "pounds sterling": "GBP",
    "₹": "INR", "indian rupees": "INR", "indian rupee": "INR",
    "yen": "JPY", "yuan": "CNY", "renminbi": "CNY",
}  # fmt: skip
# Ambiguous form (as written, lower case) -> canonical key used in configured defaults.
AMBIGUOUS_CURRENCY_FORMS: dict[str, str] = {
    "$": "$", "rs": "rs", "rs.": "rs", "rupees": "rupees", "rupee": "rupees",
    "dollars": "dollars", "dollar": "dollars", "pounds": "pounds", "pound": "pounds",
    "¥": "¥",
}  # fmt: skip
CurrencyDefaults = Mapping[str, str]


def normalize_currency_default_key(key: str) -> str:
    """'Rs.' / 'RS' / 'rs' -> 'rs'; 'Rupee' -> 'rupees'. Raises for unknown forms."""
    k = key.strip().casefold()
    canonical_key = AMBIGUOUS_CURRENCY_FORMS.get(k) or AMBIGUOUS_CURRENCY_FORMS.get(k + ".")
    if canonical_key is None:
        raise ValueError(
            f"{key!r} is not an ambiguous currency form; known: "
            f"{sorted(set(AMBIGUOUS_CURRENCY_FORMS.values()))}"
        )
    return canonical_key


@dataclass(frozen=True)
class CurrencyMention:
    start: int
    end: int
    surface: str
    code: str | None  # resolved ISO code; None if ambiguous and no default configured
    ambiguous_key: str | None  # canonical ambiguous form ("rs", "$", ...) if ambiguous


def _currency_pattern() -> re.Pattern[str]:
    words = sorted({*UNAMBIGUOUS_CURRENCY_FORMS, *AMBIGUOUS_CURRENCY_FORMS}, key=len, reverse=True)
    alts = []
    for w in words:
        e = re.escape(w)
        alts.append(rf"(?<![A-Za-z]){e}(?![A-Za-z])" if w[0].isalpha() else e)
    return re.compile("|".join(alts), re.IGNORECASE)


_CURRENCY_WORDS = _currency_pattern()
_CURRENCY_CODE = re.compile(r"(?<![A-Za-z])(" + "|".join(sorted(CURRENCY_CODES)) + r")(?![A-Za-z])")


def currency_mentions(text: str, defaults: CurrencyDefaults | None = None) -> list[CurrencyMention]:
    """Every currency mention in ``text``, in order. Overlaps resolve to the longest
    form ("US$" over "$", "Sri Lankan rupees" over "rupees")."""
    resolved = {normalize_currency_default_key(k): v for k, v in (defaults or {}).items()}
    found: list[CurrencyMention] = []
    for m in _CURRENCY_CODE.finditer(text):
        found.append(CurrencyMention(m.start(), m.end(), m.group(0), m.group(0), None))
    for m in _CURRENCY_WORDS.finditer(text):
        form = m.group(0).casefold()
        if form in UNAMBIGUOUS_CURRENCY_FORMS:
            found.append(CurrencyMention(m.start(), m.end(), m.group(0),
                                         UNAMBIGUOUS_CURRENCY_FORMS[form], None))  # fmt: skip
        else:
            key = AMBIGUOUS_CURRENCY_FORMS[form]
            found.append(CurrencyMention(m.start(), m.end(), m.group(0), resolved.get(key), key))
    found.sort(key=lambda c: (c.start, -(c.end - c.start)))
    out: list[CurrencyMention] = []
    for c in found:
        if out and c.start < out[-1].end:
            continue  # contained in a longer, earlier mention
        out.append(c)
    return out


def locate_currency(
    code: str, text: str, defaults: CurrencyDefaults | None = None
) -> tuple[int, int] | None:
    """Span of the first mention that resolves to ``code`` in ``text``."""
    for c in currency_mentions(text, defaults):
        if c.code == code.upper():
            return c.start, c.end
    return None


def currency_in(code: str, text: str, defaults: CurrencyDefaults | None = None) -> bool:
    """Whether ``text`` mentions ``code`` (ISO 4217) in a form that resolves to it.
    Ambiguous symbols ('Rs.', '$') count only through an explicit configured default."""
    return locate_currency(code, text, defaults) is not None


_ATTACH_BEFORE = re.compile(r"[\s:]*")  # "LKR 5", "Rs.5", "$5", "LKR: 5"
_ATTACH_AFTER = re.compile(r"\s*")  # "5 million LKR", "5,000 rupees"


def currencies_attached(
    text: str, start: int, end: int, defaults: CurrencyDefaults | None = None
) -> list[CurrencyMention]:
    """Currency mentions directly attached to the amount at ``text[start:end]``: the
    code or symbol immediately before it, or immediately after it, with only spaces
    (or a colon, before) in between. A currency elsewhere in the text is not attached."""
    out = []
    for c in currency_mentions(text, defaults):
        if c.end <= start and _ATTACH_BEFORE.fullmatch(text[c.end : start]):
            out.append(c)
        elif c.start >= end and _ATTACH_AFTER.fullmatch(text[end : c.start]):
            out.append(c)
    return out


# ------------------------------------------------------------------------ modality

MODALITY_PATTERNS: dict[Modality, re.Pattern[str]] = {
    Modality.PROPOSED: re.compile(r"\b(propos(?:ed|es|e|al|ing)|request(?:ed|s)?|planned)\b", re.I),
    Modality.APPROVED: re.compile(
        r"\b(approv(?:ed|al|es)|allocat(?:ed|ion)|sanction(?:ed)?|authori[sz]ed)\b", re.I
    ),
    Modality.ACTUAL: re.compile(r"\b(actual(?:ly)?|spent|recorded|incurred|disbursed)\b", re.I),
    Modality.FORECAST: re.compile(r"\b(forecast(?:ed)?|projected|expected|estimated)\b", re.I),
}  # fmt: skip

MODALITY_WORDS = frozenset(
    {"proposed", "approved", "actual", "forecast", "forecasted", "projected", "estimated",
     "planned", "requested", "sanctioned", "allocated"}
)  # fmt: skip
_WORD_MODALITY = {
    "proposed": Modality.PROPOSED, "planned": Modality.PROPOSED, "requested": Modality.PROPOSED,
    "approved": Modality.APPROVED, "sanctioned": Modality.APPROVED, "allocated": Modality.APPROVED,
    "actual": Modality.ACTUAL,
    "forecast": Modality.FORECAST, "forecasted": Modality.FORECAST,
    "projected": Modality.FORECAST, "estimated": Modality.FORECAST,
}  # fmt: skip


def modalities_in(text: str) -> set[Modality]:
    return {mod for mod, pat in MODALITY_PATTERNS.items() if pat.search(text)}


# ----------------------------------------------------------------------- predicates

PREDICATE_SYNONYMS: dict[str, frozenset[str]] = {
    "budget": frozenset({"budget", "allocation", "budget_allocation", "allocated_budget",
                         "funding", "funding_allocation", "grant", "total_budget",
                         "budget_amount", "allocated_amount", "budget_request", "requested_budget",
                         "requested_amount"}),
    "expenditure": frozenset({"expenditure", "spending", "spend", "expense", "expenses",
                              "cost_incurred", "amount_spent", "total_expenditure", "spent"}),
    "cost": frozenset({"cost", "item_cost", "estimated_cost", "cost_estimate"}),
    "completion_date": frozenset({"completion_date", "end_date", "completed_on", "date_completed",
                                  "completed", "completion"}),
    "start_date": frozenset({"start_date", "started_on", "commencement_date", "start"}),
    "status": frozenset({"status", "project_status", "milestone", "milestone_status"}),
    "supervisor": frozenset({"supervisor", "supervised_by", "supervision", "project_supervisor"}),
    "host_institution": frozenset({"host_institution", "host", "hosted_by", "host_organization"}),
    "owner": frozenset({"owner", "project_owner", "owned_by", "responsible_party"}),
}  # fmt: skip
_CANONICAL = {syn: canon for canon, syns in PREDICATE_SYNONYMS.items() for syn in syns}


def snake_case(text: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", text.strip()).strip("_").lower()
    s = re.sub(r"_+", "_", s)
    return s if s and s[0].isalpha() else f"p_{s}" if s else ""


def normalize_predicate(raw: str) -> tuple[str, Modality | None, bool]:
    """('Approved Budget') -> ('budget', Modality.APPROVED, True).

    Modality words are moved out of the predicate, so "approved budget" and
    "proposed budget" share the predicate ``budget`` and differ only in modality.
    Returns (predicate, modality implied by the removed words, in_vocabulary).
    """
    tokens = snake_case(raw).split("_")
    implied = {_WORD_MODALITY[t] for t in tokens if t in _WORD_MODALITY}
    kept = [t for t in tokens if t not in MODALITY_WORDS]
    base = "_".join(kept) or "value"
    canonical = _CANONICAL.get(base)
    implied_mod = implied.pop() if len(implied) == 1 else None
    return (canonical or base), implied_mod, canonical is not None


# -------------------------------------------------------------------------- dates
#
# A date is grounded only by a single date *expression* in the text that states
# the same year, month and day (as far as the claimed date specifies them). A
# month name elsewhere in the text never grounds a month for a different mention
# of the year.

_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august",
           "september", "october", "november", "december"]  # fmt: skip
_MONTH_NUM = {name: i + 1 for i, name in enumerate(_MONTHS)}
_MONTH_NUM.update({name[:3]: i + 1 for i, name in enumerate(_MONTHS)})
_MONTH_NUM["sept"] = 9
_MON = r"(?P<mon>" + "|".join(sorted(_MONTH_NUM, key=len, reverse=True)) + r")\.?"
_DAY = r"(?P<{name}>[0-3]?\d)(?:st|nd|rd|th)?"
_YEAR = r"(?P<year>[12]\d{3})"

_DATE_PATTERNS: list[re.Pattern[str]] = [
    # 2025-01-15, 2025/01/15, 2025.01.15, 2025-01
    re.compile(rf"(?<![\d.,/-]){_YEAR}[-/.](?P<m>[01]?\d)(?:[-/.](?P<d>[0-3]?\d))?(?!\d|,\d)"),
    # 15/01/2025 or 01/15/2025 (day/month order resolved below), 01/2025
    re.compile(rf"(?<![\d.,/-])(?:(?P<a>[0-3]?\d)[-/.])?(?P<b>[01]?\d)[-/.]{_YEAR}(?!\d|,\d)"),
    # 15 January 2025, 15th of Jan. 2025, January 2025
    re.compile(rf"\b(?:{_DAY.format(name='d1')}\s+(?:of\s+)?)?{_MON}\s*,?\s+{_YEAR}\b", re.I),
    # January 15, 2025 / Jan 15 2025
    re.compile(rf"\b{_MON}\s+{_DAY.format(name='d2')},?\s+{_YEAR}\b", re.I),
]
_BARE_YEAR = re.compile(r"(?<![\d.,])[12]\d{3}(?!\d|,\d)")

DateMention = tuple[int, int | None, int | None]


def _valid(y: int, m: int | None, d: int | None) -> bool:
    try:
        date(y, 1 if m is None else m, 1 if d is None else d)
    except ValueError:
        return False
    return True


# Date *shapes*, independent of calendar validity. Used only for masking: the digits
# of anything shaped like a date ("2025-99-99", "31/02/2025", "2025-13-40") are not
# quantities, valid or not. Each shape is matched as a whole token (no digit, dot,
# comma, slash or hyphen directly before or after), so a mask never covers half of
# a token and never reaches into a neighbouring number.
_TOKEN_START = r"(?<![\d.,/-])"
_TOKEN_END = r"(?![\d/-]|[.,]\d)"
# A dot is accepted only in three-part shapes: "5.2025" or "2025.05" are decimals.
_DATE_SHAPES: list[re.Pattern[str]] = [
    re.compile(_TOKEN_START + r"[12]\d{3}([-/.])\d{1,2}\1\d{1,2}" + _TOKEN_END),  # Y-M-D
    re.compile(_TOKEN_START + r"\d{1,2}([-/.])\d{1,2}\1[12]\d{3}" + _TOKEN_END),  # D/M/Y, M/D/Y
    re.compile(_TOKEN_START + r"[12]\d{3}[-/]\d{1,2}" + _TOKEN_END),  # Y-M
    re.compile(_TOKEN_START + r"\d{1,2}[-/][12]\d{3}" + _TOKEN_END),  # M/Y
]


def date_spans(text: str) -> list[tuple[int, int]]:
    """Character ranges of date-shaped expressions: numeric shapes (whether or not
    they are real calendar dates) and month-name dates. Bare years are excluded."""
    spans = [m.span() for pat in _DATE_SHAPES for m in pat.finditer(text)]
    spans += [m.span() for pat in _DATE_PATTERNS[2:] for m in pat.finditer(text)]  # month names
    return sorted(spans)


def mask_dates(text: str) -> str:
    """``text`` with every date-shaped expression replaced by spaces (offsets kept).

    Invalid dates are masked too ("2025-99-99" yields no quantities 99), while
    numbers outside the date token are untouched ("Paid 2025-99-99 total 5,000" ->
    5000). Calendar validity matters only for grounding a claimed date
    (``date_mentions`` / ``date_in``), never for whether digits count as quantities.
    """
    chars = list(text)
    for a, b in date_spans(text):
        chars[a:b] = " " * (b - a)
    return "".join(chars)


def date_mentions(text: str) -> set[DateMention]:
    """Every (year, month, day) the text states unambiguously; month/day may be None.

    Numeric dates whose day/month order is ambiguous (e.g. 01/02/2025) contribute
    only their year, never a guessed month or day.
    """
    found: set[DateMention] = set()
    iso, numeric, day_first, month_first = _DATE_PATTERNS
    for mt in iso.finditer(text):
        y, mo = int(mt["year"]), int(mt["m"])
        d = int(mt["d"]) if mt["d"] else None
        if _valid(y, mo, d):
            found.add((y, mo, d))
    for mt in numeric.finditer(text):
        y, b = int(mt["year"]), int(mt["b"])
        if mt["a"] is None:
            if _valid(y, b, None):
                found.add((y, b, None))
            continue
        a = int(mt["a"])
        readings = {(y, mo, d) for mo, d in ((b, a), (a, b)) if _valid(y, mo, d)}
        if len(readings) == 1:
            found |= readings
    for mt in day_first.finditer(text):
        y, mo = int(mt["year"]), _MONTH_NUM[mt["mon"].lower()]
        d = int(mt["d1"]) if mt["d1"] else None
        if _valid(y, mo, d):
            found.add((y, mo, d))
    for mt in month_first.finditer(text):
        y, mo, d = int(mt["year"]), _MONTH_NUM[mt["mon"].lower()], int(mt["d2"])
        if _valid(y, mo, d):
            found.add((y, mo, d))
    for mt in _BARE_YEAR.finditer(text):
        found.add((int(mt.group(0)), None, None))
    return found


def is_partial_date(value: str) -> bool:
    """YYYY, YYYY-MM or YYYY-MM-DD and a real calendar date (rejects 2025-02-30)."""
    return parse_partial_date(value) is not None


def date_in(value: str, text: str) -> bool:
    """Whether ``text`` states the date ``value`` at its own precision.

    YYYY needs that year; YYYY-MM needs one expression with that year *and* month;
    YYYY-MM-DD needs one expression with that exact day, month and year.
    """
    parsed = parse_partial_date(value)
    if parsed is None:
        return False
    year, month, day = parsed
    for y, m, d in date_mentions(text):
        if y != year:
            continue
        if month is None:
            return True
        if m == month and (day is None or d == day):
            return True
    return False


# ------------------------------------------------------------------------- booleans
#
# A boolean claim ("Project Alpha | completed | true") is grounded only by a clause
# that contains BOTH the subject and the predicate's content word. Polarity comes
# from explicit negation cues in that clause before the predicate word. Hedged,
# conditional, interrogative or doubly negated clauses do not establish a value.

Polarity = Literal["true", "false", "ambiguous", "absent"]

_NEGATIONS = frozenset({"not", "no", "never", "neither", "nor", "without", "none", "cannot"})
_NEG_PHRASES = re.compile(r"\b(failed to|yet to|has not|have not|did not|was not|is not)\b", re.I)
_HEDGES = frozenset({"may", "might", "could", "would", "should", "will", "expected", "planned",
                     "likely", "possibly", "perhaps", "if", "whether", "unless", "proposed",
                     "intended", "scheduled", "aims", "aim"})  # fmt: skip
_PRED_STOP = frozenset({"is", "has", "have", "was", "were", "be", "been", "are", "the", "of",
                        "a", "an", "flag", "status", "value", "p"})  # fmt: skip
# Clauses break at sentence punctuation, commas and contrastive conjunctions, so
# "Unlike Project Alpha, Project Beta was completed" never grounds Project Alpha.
_CLAUSE_SPLIT = re.compile(
    r"(?<=[.;:!?])\s+|\s*,\s*|\s*\b(?:but|however|whereas|although|while|unlike)\b\s*",
    re.I,
)  # fmt: skip
YES_VALUES = frozenset({"yes", "y", "true", "✓", "✔", "x"})
NO_VALUES = frozenset({"no", "n", "false", "✗", "✘", "-"})


def _stem(word: str) -> str:
    w = word.lower().strip("'’\"()[],")
    for suffix in ("ation", "ions", "ion", "ing", "ed", "es", "s", "e"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            return w[: -len(suffix)]
    return w


def predicate_stems(predicate: str) -> set[str]:
    return {_stem(t) for t in predicate.split("_") if t not in _PRED_STOP and len(t) >= 3}


def boolean_polarity(text: str, subject: str, predicate: str) -> Polarity:
    stems = predicate_stems(predicate)
    if not stems:
        return "absent"
    verdicts: set[Polarity] = set()
    for clause in _CLAUSE_SPLIT.split(text):
        subj = locate(subject, clause) if clause else None
        if subj is None:
            continue
        words = re.findall(r"[\w'’]+|\?", clause)
        hits = [i for i, w in enumerate(words) if _stem(w) in stems]
        if not hits:
            continue
        # The subject must precede the predicate word in the clause ("Project Alpha
        # was completed"), so a trailing mention cannot inherit another's predicate.
        keyword_at = [m.start() for m in re.finditer(r"[\w'’]+|\?", clause)][hits[0]]
        if subj[1] > keyword_at:
            continue
        before = [w.lower() for w in words[: hits[0]]]
        if clause.rstrip().endswith("?") or any(w in _HEDGES for w in before):
            verdicts.add("ambiguous")
            continue
        negs = sum(w in _NEGATIONS or w.endswith(("n't", "n’t")) for w in before)
        negs = max(negs, len(_NEG_PHRASES.findall(" ".join(before))))
        verdicts.add("true" if negs == 0 else "false" if negs == 1 else "ambiguous")
    if not verdicts:
        return "absent"
    return verdicts.pop() if len(verdicts) == 1 else "ambiguous"


def header_names_predicate(label: str, predicate: str) -> bool:
    stems = predicate_stems(predicate)
    return bool(stems & {_stem(w) for w in re.findall(r"[\w'’]+", label)})


def cell_polarity(label: str, value: str, predicate: str) -> Polarity:
    """Table cell under a header naming the predicate, e.g. 'Completed' = 'Yes'."""
    if not header_names_predicate(label, predicate):
        return "absent"
    v = value.strip().lower()
    return "true" if v in YES_VALUES else "false" if v in NO_VALUES else "ambiguous"


# ------------------------------------------------------------------- sentences
#
# Grounding scope for prose: a subject, text value, unit-bearing phrase or time
# expression may support a claim only if it is in the same sentence as one of
# the claim's quotes. Another sentence of the same paragraph is a different
# statement, possibly about a different entity.

_ABBREVIATIONS = frozenset(
    {
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "no",
        "st",
        "vs",
        "etc",
        "inc",
        "ltd",
        "co",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "rs",
        "e.g",
        "i.e",
        "approx",
        "dept",
    }
)
_SENTENCE_END = re.compile(r"[.!?;](?=\s+|$)|\n\s*\n")


def sentence_starts(text: str) -> list[int]:
    """Offsets where sentences begin (always includes 0)."""
    starts = [0]
    for m in _SENTENCE_END.finditer(text):
        if m.group(0) == ".":
            before = re.search(r"([\w.]+)$", text[: m.start()])
            word = before.group(1).lower() if before else ""
            if word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
                continue  # "Dr." or an initial such as "A."
        nxt = re.compile(r"\s*").match(text, m.end())
        pos = nxt.end() if nxt else m.end()
        if pos < len(text):
            starts.append(pos)
    return sorted(set(starts))


def sentence_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    """Smallest run of whole sentences covering text[start:end]."""
    starts = sentence_starts(text)
    s0 = max(s for s in starts if s <= start)
    later = [s for s in starts if s >= end]
    s1 = min(later) if later else len(text)
    return s0, s1
