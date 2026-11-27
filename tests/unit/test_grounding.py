from __future__ import annotations

from decimal import Decimal

import pytest

from veritas.core.contracts import Modality
from veritas.extraction import grounding as g


@pytest.mark.parametrize(
    ("quote", "text", "expected"),
    [
        ("LKR 5 million", "an allocation of LKR 5 million in 2025", (17, 30)),
        ("lkr 5   million", "an allocation of LKR 5\nmillion in 2025", (17, 30)),
        ("LKR 7 million", "an allocation of LKR 5 million", None),
        ("", "anything", None),
    ],
)
def test_locate(quote: str, text: str, expected: tuple[int, int] | None) -> None:
    assert g.locate(quote, text) == expected


def test_locate_returns_original_characters() -> None:
    text = "Approved Allocation of LKR 5 Million"
    start, end = g.locate("approved allocation", text)  # type: ignore[misc]
    assert text[start:end] == "Approved Allocation"


@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        ("LKR 5 million", [Decimal("5000000")]),
        ("5,500,000", [Decimal("5500000")]),
        ("4.5M and 2.5bn", [Decimal("4500000"), Decimal("2500000000")]),
        ("Rs. 3 lakh", [Decimal("300000")]),
        ("in 2025", [Decimal("2025")]),
        ("version 1.2.3", [Decimal("1.2")]),  # '.3' is not read as a separate number
    ],
)
def test_parse_numbers(text: str, numbers: list[Decimal]) -> None:
    assert g.parse_numbers(text) == [n.normalize() for n in numbers]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("5000000", "5000000"),
        ("5,000,000", "5000000"),
        ("LKR 5,000,000", "5000000"),
        ("5 million", "5000000"),
        ("12.50", "12.5"),
        ("abc", None),
    ],
)
def test_parse_value(value: str, expected: str | None) -> None:
    result = g.parse_value(value)
    assert result == (Decimal(expected).normalize() if expected else None)


def test_number_in() -> None:
    assert g.number_in(Decimal("5000000"), "approved allocation of LKR 5 million")
    assert not g.number_in(Decimal("7000000"), "approved allocation of LKR 5 million")


@pytest.mark.parametrize(
    ("code", "text", "ok"),
    [
        ("LKR", "Amount (LKR) = 5500000", True),
        ("LKR", "Rs. 5,000", False),  # 'Rs.' is ambiguous without a configured default
        ("LKR", "USD 5,000", False),
        ("USD", "US$ 20", True),
        ("LKR", "BULKRATE", False),
    ],
)
def test_currency_in(code: str, text: str, ok: bool) -> None:
    assert g.currency_in(code, text) is ok


@pytest.mark.parametrize(
    ("text", "mods"),
    [
        ("The team proposes a budget of LKR 7 million", {Modality.PROPOSED}),
        ("received an approved allocation of LKR 5 million", {Modality.APPROVED}),
        ("recorded actual expenditure of LKR 4,500,000", {Modality.ACTUAL}),
        ("forecast expenditure is LKR 6 million", {Modality.FORECAST}),
        ("total budget is LKR 6 million", set()),
    ],
)
def test_modalities_in(text: str, mods: set[Modality]) -> None:
    assert g.modalities_in(text) == mods


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("approved_budget", ("budget", Modality.APPROVED, True)),
        ("Proposed Budget", ("budget", Modality.PROPOSED, True)),
        ("budget_allocation", ("budget", None, True)),
        ("actual_expenditure", ("expenditure", Modality.ACTUAL, True)),
        ("amount spent", ("expenditure", None, True)),
        ("number_of_staff", ("number_of_staff", None, False)),
    ],
)
def test_normalize_predicate(raw: str, expected: tuple[str, Modality | None, bool]) -> None:
    assert g.normalize_predicate(raw) == expected


def test_budget_semantics_stay_distinct() -> None:
    """proposed/approved budget share a predicate; actual expenditure does not."""
    proposed = g.normalize_predicate("proposed_budget")
    approved = g.normalize_predicate("approved_budget")
    actual = g.normalize_predicate("actual_expenditure")
    assert proposed[0] == approved[0] == "budget"
    assert proposed[1] != approved[1]
    assert actual[0] == "expenditure"


@pytest.mark.parametrize(
    ("value", "text", "ok"),
    [
        ("2025", "in January 2025", True),
        ("2025-01", "in January 2025", True),
        ("2025-01", "Jan. 2025", True),
        ("2025-01", "2025-01", True),
        ("2025-06", "in January 2025", False),
        ("2026", "in January 2025", False),
        ("2025-03", "summary for 2025", False),
    ],
)
def test_date_in(value: str, text: str, ok: bool) -> None:
    assert g.date_in(value, text) is ok


def test_whole_numbers_have_no_exponent() -> None:
    v = g.parse_value("5,000,000")
    assert v is not None and str(v) == "5000000"
    assert str(g.parse_numbers("LKR 2.5 million")[0]) == "2500000"
    assert str(g.parse_value("12.50")) == "12.5"


# ----------------------------------------------------------- dates (corrective)


@pytest.mark.parametrize(
    ("value", "text", "ok"),
    [
        # YYYY: the exact year must be stated
        ("2025", "completed in 2025", True),
        ("2025", "completed in 2024", False),
        ("2025", "budget of LKR 2025000", False),  # digits inside a larger number
        ("2025", "on 15/01/2025", True),
        # YYYY-MM: year and month in the SAME expression
        ("2025-06", "completed in June 2025", True),
        ("2025-06", "completed in Jun. 2025", True),
        ("2025-06", "2025-06", True),
        ("2025-06", "2025/6", True),
        ("2025-06", "06/2025", True),
        ("2025-06", "on 2025-06-30", True),
        ("2025-06", "completed in July 2025", False),
        ("2025-06", "completed in June 2024", False),
        ("2025-06", "June 2024 and later 2025", False),  # month borrowed from another date
        ("2025-01", "report of January, results for 2025", False),
        # YYYY-MM-DD: exact calendar day
        ("2025-01-15", "on 15 January 2025", True),
        ("2025-01-15", "on 15th of January 2025", True),
        ("2025-01-15", "on January 15, 2025", True),
        ("2025-01-15", "on 2025-01-15", True),
        ("2025-01-15", "on 15/01/2025", True),  # only one valid day/month reading
        ("2025-01-15", "on 16 January 2025", False),
        ("2025-01-15", "in January 2025", False),  # month precision only
        ("2025-02-01", "on 01/02/2025", False),  # ambiguous: 1 Feb or 2 Jan
        # leap years and impossible dates
        ("2024-02-29", "on 29 February 2024", True),
        ("2025-02-29", "on 29 February 2025", False),
        ("2025-02-30", "on 2025-02-30", False),
    ],
)
def test_date_grounding(value: str, text: str, ok: bool) -> None:
    assert g.date_in(value, text) is ok


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ("2025", True),
        ("2025-02", True),
        ("2024-02-29", True),
        ("2025-02-29", False),
        ("2025-02-30", False),
        ("2025-13", False),
        ("2025-00", False),
        ("2025-15", False),
        ("2025-04-31", False),
        ("25-01", False),
        ("2025-1", False),
    ],
)
def test_partial_date_calendar_validation(value: str, valid: bool) -> None:
    assert g.is_partial_date(value) is valid


def test_text_with_impossible_numeric_date_grounds_nothing_but_year() -> None:
    assert g.date_mentions("filed 31/02/2025") == {(2025, None, None)}


# ---------------------------------------------------------------- booleans


@pytest.mark.parametrize(
    ("text", "subject", "predicate", "polarity"),
    [
        ("Project Alpha was completed in June 2025.", "Project Alpha", "completed", "true"),
        ("Project Alpha was not completed in 2025.", "Project Alpha", "completed", "false"),
        ("Project Alpha wasn't completed.", "Project Alpha", "is_completed", "false"),
        ("Project Alpha has never been approved.", "Project Alpha", "approved", "false"),
        ("Project Alpha may be completed in 2026.", "Project Alpha", "completed", "ambiguous"),
        ("Project Alpha is expected to be completed.", "Project Alpha", "completed", "ambiguous"),
        ("Was Project Alpha completed?", "Project Alpha", "completed", "ambiguous"),
        (
            "It is not true that Project Alpha was not completed.",
            "Project Alpha",
            "completed",
            "ambiguous",
        ),
        (
            "Project Beta was completed, but Project Alpha was not.",
            "Project Alpha",
            "completed",
            "absent",
        ),
        ("Project Alpha was approved.", "Project Alpha", "completed", "absent"),
        ("Project Beta was completed.", "Project Alpha", "completed", "absent"),
    ],
)
def test_boolean_polarity(text: str, subject: str, predicate: str, polarity: str) -> None:
    assert g.boolean_polarity(text, subject, predicate) == polarity


@pytest.mark.parametrize(
    ("label", "value", "polarity"),
    [
        ("Completed", "Yes", "true"),
        ("Completed?", "No", "false"),
        ("Completed", "Partly", "ambiguous"),
        ("Owner", "Yes", "absent"),
    ],
)
def test_cell_polarity(label: str, value: str, polarity: str) -> None:
    assert g.cell_polarity(label, value, "completed") == polarity


@pytest.mark.parametrize("text", ["2025-00", "00/2025", "0 January 2025"])
def test_zero_month_or_day_is_never_a_date(text: str) -> None:
    assert all(m in (None, 0) or 1 <= m <= 12 for _, m, _ in g.date_mentions(text))
    assert not g.date_in("2025-01", text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Project Alpha started. Project Beta ended.",
            ["Project Alpha started. ", "Project Beta ended."],
        ),
        ("Supervised by Dr. A. Perera. Done.", ["Supervised by Dr. A. Perera. ", "Done."]),
        ("Budget approved; work began", ["Budget approved; ", "work began"]),
    ],
)
def test_sentence_bounds(text: str, expected: list[str]) -> None:
    starts = g.sentence_starts(text)
    pieces = [text[a:b] for a, b in zip(starts, [*starts[1:], len(text)], strict=True)]
    assert pieces == expected


# ------------------------------------------- numbers exclude date components


@pytest.mark.parametrize(
    "text",
    [
        "The project was completed on 2025-01-15.",
        "The project was completed on 2025/01/15.",
        "The project was completed on 15/01/2025.",
        "The project was completed on 15 January 2025.",
        "The project was completed on January 15, 2025.",
        "The project was completed on 15th of Jan. 2025.",
    ],
)
def test_date_components_are_not_quantities(text: str) -> None:
    for part in ("15", "1", "2025", "2025.01"):
        assert not g.number_in(Decimal(part), text), (part, text)
    assert g.parse_numbers(text) == []


@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        ("15 projects completed on 2025-01-15", ["15"]),
        ("On 15/01/2025, 12 sites opened", ["12"]),
        ("LKR 5,500,000 approved on 15 January 2025", ["5500000"]),
        ("4.5 million spent by June 2025", ["4500000"]),
        ("Rs. 3 lakh and 2 crore in March 2025", ["300000", "20000000"]),
        ("2.5bn and 7 thousand (2025-02-01)", ["2500000000", "7000"]),
        ("40% done on 2025-06-30", ["40"]),
    ],
)
def test_quantities_next_to_dates_are_kept(text: str, numbers: list[str]) -> None:
    assert g.parse_numbers(text) == [Decimal(n) for n in numbers]


def test_bare_year_is_still_a_number() -> None:
    """Documented limitation: 'in 2025' cannot be distinguished from a count."""
    assert g.parse_numbers("completed in 2025") == [Decimal("2025")]


@pytest.mark.parametrize(
    ("value", "text"),
    [
        ("2025-01-15", "On 15/01/2025, 12 sites opened"),
        ("2025-01-15", "On 2025-01-15, work began"),
        ("2025", "In 2025, work began"),
    ],
)
def test_date_followed_by_comma_is_recognised(value: str, text: str) -> None:
    assert g.date_in(value, text)


def test_year_followed_by_thousands_group_is_not_a_date() -> None:
    assert g.parse_numbers("LKR 2,025,000") == [Decimal("2025000")]
    assert not g.date_in("2025", "LKR 2,025,000")


# ---------------------------------------------- invalid date-shaped strings


@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        ("2025-99-99", []),
        ("2025-02-30", []),
        ("31/02/2025", []),
        ("2025-13-40", []),  # previously half-masked, leaving a spurious 40
        ("12/99/2025", []),
        ("99/2025", []),
        ("2025.02.30", []),
    ],
)
def test_invalid_date_shapes_yield_no_quantities(text: str, numbers: list[str]) -> None:
    assert g.parse_numbers(text) == [Decimal(n) for n in numbers]
    assert g.mask_dates(text).strip() == ""


@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        ("Paid 2025-99-99 total 5,000", ["5000"]),
        ("On 31/02/2025 we spent LKR 5 million", ["5000000"]),
        ("Invoice 2025-13-40: 12 items", ["12"]),
        ("ref 2025/02/30 amount 300", ["300"]),
        ("12/99/2025 and 7 units", ["7"]),
        ("99/2025 count 4", ["4"]),
        ("2025-02-30, 1.5 million", ["1500000"]),
    ],
)
def test_invalid_dates_never_hide_neighbouring_numbers(text: str, numbers: list[str]) -> None:
    assert g.parse_numbers(text) == [Decimal(n) for n in numbers]


@pytest.mark.parametrize(
    ("text", "numbers"),
    [
        ("5.2025 million", ["5202500"]),  # a decimal, not "May 2025"
        ("2025.05 tonnes", ["2025.05"]),  # a decimal, not "2025-05"
        ("LKR 12.2025", ["12.2025"]),
        ("ID 2025-0042 qty 3", ["2025", "42", "3"]),  # not date-shaped
        ("2024-2025: 40 staff", ["2024", "2025", "40"]),  # a year range, not a date
    ],
)
def test_numbers_that_merely_resemble_dates_are_kept(text: str, numbers: list[str]) -> None:
    assert g.parse_numbers(text) == [Decimal(n) for n in numbers]


def test_masking_never_crosses_token_boundaries() -> None:
    text = "Invoice 2025-13-40: 12 items"
    masked = g.mask_dates(text)
    assert len(masked) == len(text)
    assert masked[: text.index("2025")] == text[: text.index("2025")]
    assert masked[text.index(":") :] == text[text.index(":") :]


def test_invalid_dates_ground_no_month_or_day() -> None:
    """Intended: an invalid date grounds no month or day. Its year still counts as a
    bare-year mention (existing, documented rule)."""
    for text in ("on 2025-02-30", "on 31/02/2025", "on 2025-99-99"):
        assert not g.date_in("2025-02", text)
        assert not g.date_in("2025-02-28", text)
        assert g.date_in("2025", text)
