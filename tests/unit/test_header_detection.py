"""Conservative table-header detection: a row is only ever dropped from the data
when it is positively identified as a header; uncertainty keeps every row."""

from __future__ import annotations

from typing import Any

import pytest
from factories import llm_claim, xlsx_table

from veritas.core.contracts import DocumentFormat, DocumentSpan
from veritas.extraction.schema import LLMClaim
from veritas.extraction.sources import (
    HeaderStatus,
    SourcePlan,
    detect_header_row,
    is_label_like,
    plan_sources,
)
from veritas.extraction.validator import ClaimValidator

DOC = "DOC-000000000001"


def cells(rows: list[list[str]], drop: set[tuple[int, int]] = frozenset()) -> list[DocumentSpan]:  # type: ignore[assignment]
    out = xlsx_table(DOC, "S", rows)
    return [
        c
        for c in out
        if c.location and (c.location.table_row, c.location.table_col) not in drop and c.text != ""
    ]


def plan(rows: list[list[str]], **kw: Any) -> SourcePlan:
    return plan_sources(cells(rows, **kw), DocumentFormat.XLSX)


def texts(p: SourcePlan) -> list[str]:
    return [u.text for b in p.batches for u in b]


def status(rows: list[list[str]], **kw: Any) -> HeaderStatus:
    by_row: dict[int, dict[int, DocumentSpan]] = {}
    for c in cells(rows, **kw):
        assert c.location and c.location.table_row is not None
        by_row.setdefault(c.location.table_row, {})[c.location.table_col or 0] = c
    return detect_header_row(by_row)


def assert_no_cell_lost(rows: list[list[str]], p: SourcePlan) -> None:
    """Every input cell is either data in some unit or (only for HEADER) a header."""
    data = {sp.span_id for b in p.batches for u in b for sp in u.spans}
    hdrs = {h.span_id for b in p.batches for u in b for h in u.header_spans if h}
    for c in cells(rows):
        assert c.span_id in data | hdrs, f"cell {c.text!r} disappeared"
    if p.tables_with_header == 0:
        assert not hdrs


# ------------------------------------------------------ confirmed reproduction


def test_review_reproduction_keeps_both_rows_as_data() -> None:
    rows = [["Project Alpha", "pending"], ["Project Beta", "5"]]
    p = plan(rows)
    assert texts(p) == ["Project Alpha | pending", "Project Beta | 5"]
    assert (p.tables_with_header, p.tables_without_header, p.tables_header_ambiguous) == (0, 0, 1)
    assert status(rows) is HeaderStatus.AMBIGUOUS
    assert_no_cell_lost(rows, p)


def test_ambiguous_first_row_never_supplies_labels_or_currency() -> None:
    """Row 0 'LKR' is not a recognised label: it must not qualify the value below."""
    rows = [["Project Alpha", "LKR"], ["Project Beta", "5"]]
    p = plan(rows)
    assert texts(p) == ["Project Alpha | LKR", "Project Beta | 5"]
    beta = p.batches[0][1]
    assert all(h is None for h in beta.header_spans)
    ec = ClaimValidator(p.batches[0], DOC, "RUN-t").validate_claim(
        LLMClaim.model_validate(
            llm_claim(
                source_id=beta.source_id,
                subject="Project Beta",
                value="5",
                modality="unknown",
                evidence_quotes=["Project Beta", "5"],
            )
        )
    )
    assert "currency_not_grounded" in ec.issues
    assert not [e for e in ec.claim.evidence if e.role == "header"]


# ------------------------------------------- mixed rows: header or data?


@pytest.mark.parametrize(
    "rows",
    [
        [["Project Alpha", "pending", "note"], ["Project Beta", "5", "x"]],
        [["Phase", "Ongoing"], ["Alpha", "5"]],  # 'Ongoing' is not a label
        [["Project Alpha (Phase 2)", "Amount"], ["Project Beta", "5"]],
        [["Kandy", "Budget"], ["Colombo", "5000000"]],  # a place name in row 0
        [["Status", "Amount"], ["Status", "5"]],  # row 0 repeats a value of its column
    ],
)
def test_uncertain_first_row_is_kept_as_data(rows: list[list[str]]) -> None:
    p = plan(rows)
    assert p.tables_with_header == 0
    assert len(texts(p)) == len(rows)
    assert texts(p)[0].startswith(rows[0][0])
    assert_no_cell_lost(rows, p)


def test_value_in_first_row_means_no_header() -> None:
    rows = [["Project Alpha", "5500000"], ["Project Beta", "2500000"]]
    assert status(rows) is HeaderStatus.NO_HEADER
    assert texts(plan(rows)) == ["Project Alpha | 5500000", "Project Beta | 2500000"]


def test_missing_row_zero_is_never_a_header() -> None:
    """If row 0 is absent (e.g. unlocated cells), row 1 is data, whatever it looks like."""
    rows = [["Project", "Amount"], ["Amount", "Total"], ["Project Alpha", "5"]]
    p = plan(rows, drop={(0, 0), (0, 1)})
    assert p.tables_with_header == 0
    assert texts(p) == ["Amount | Total", "Project Alpha | 5"]


# -------------------------------------------------- genuine headered tables


@pytest.mark.parametrize(
    ("rows", "first_unit"),
    [
        (
            [
                ["Project", "Modality", "Amount (LKR)", "Date"],
                ["Project Alpha", "approved", "5500000", "2025-01"],
            ],
            "Project = Project Alpha | Modality = approved | Amount (LKR) = 5500000"
            " | Date = 2025-01",
        ),
        (
            [["Project", "Milestone", "Date"], ["Project Alpha", "Completed", "2025-09"]],
            "Project = Project Alpha | Milestone = Completed | Date = 2025-09",
        ),
        (
            [["Item", "Proposed cost (LKR)"], ["Equipment", "4,000,000"]],
            "Item = Equipment | Proposed cost (LKR) = 4,000,000",
        ),
        (
            [
                ["Project", "Approved budget (LKR)", "Status"],
                ["Project Alpha", "5,000,000", "Completed"],
            ],
            "Project = Project Alpha | Approved budget (LKR) = 5,000,000 | Status = Completed",
        ),
        (
            [["Item", "Proposed cost", "Actual cost"], ["Staff", "3000000", "2800000"]],
            "Item = Staff | Proposed cost = 3000000 | Actual cost = 2800000",
        ),
        (
            [["Project", "Completed"], ["Project Alpha", "No"]],
            "Project = Project Alpha | Completed = No",
        ),
        (
            [["Region", "Beneficiaries (persons)"], ["Western", "1200"]],
            "Region = Western | Beneficiaries (persons) = 1200",
        ),
    ],
)
def test_genuine_headers_are_detected(rows: list[list[str]], first_unit: str) -> None:
    p = plan(rows)
    assert p.tables_with_header == 1
    assert texts(p)[0] == first_unit
    assert_no_cell_lost(rows, p)


def test_unknown_label_word_falls_back_to_ambiguous() -> None:
    """Documented limitation: 'Ward' is not in the label vocabulary and has no unit, so
    the table is kept whole rather than guessed."""
    rows = [["Ward", "Beneficiaries"], ["Ward 5", "1200"]]
    p = plan(rows)
    assert p.tables_header_ambiguous == 1
    assert texts(p) == ["Ward | Beneficiaries", "Ward 5 | 1200"]


# ------------------------------------------------------ headerless tables


@pytest.mark.parametrize(
    "rows",
    [
        [["Project", "Status"], ["Project Alpha", "Active"]],
        [["Project Alpha", "Active"], ["Project Beta", "Closed"]],
        [["Alpha", "Kandy", "Ongoing"], ["Beta", "Colombo", "Done"], ["Gamma", "Galle", "Hold"]],
    ],
)
def test_all_text_tables_keep_every_row(rows: list[list[str]]) -> None:
    p = plan(rows)
    assert p.tables_with_header == 0 and status(rows) is HeaderStatus.NO_HEADER
    assert len(texts(p)) == len(rows)
    assert_no_cell_lost(rows, p)


# ---------------------------------------------------------- irregular rows


def test_headered_table_with_missing_cells_keeps_all_data() -> None:
    rows = [
        ["Project", "Amount (LKR)", "Date"],
        ["Project Alpha", "5500000", ""],  # missing date
        ["Project Beta", "", "2025-03"],  # missing amount
        ["Project Gamma", "100", "2025-04"],
    ]
    p = plan(rows)
    assert p.tables_with_header == 1
    assert texts(p) == [
        "Project = Project Alpha | Amount (LKR) = 5500000",
        "Project = Project Beta | Date = 2025-03",
        "Project = Project Gamma | Amount (LKR) = 100 | Date = 2025-04",
    ]
    assert_no_cell_lost(rows, p)


def test_header_shorter_than_data_rows() -> None:
    rows = [["Project", "Amount"], ["Project Alpha", "5", "extra"]]
    p = plan(rows)
    assert p.tables_with_header == 1
    assert texts(p) == ["Project = Project Alpha | Amount = 5 | extra"]
    assert_no_cell_lost(rows, p)


def test_irregular_table_with_uncertain_header_keeps_everything() -> None:
    rows = [["Project Alpha", "pending"], ["Project Beta"], ["Project Gamma", "7", "late"]]
    p = plan(rows, drop=set())
    assert p.tables_with_header == 0
    assert texts(p) == ["Project Alpha | pending", "Project Beta", "Project Gamma | 7 | late"]
    assert_no_cell_lost(rows, p)


# -------------------------------------------------------------- single row


@pytest.mark.parametrize("row", [["Project", "Amount (LKR)"], ["Project Alpha", "5500000"]])
def test_single_row_table_is_data(row: list[str]) -> None:
    p = plan([row])
    assert status([row]) is HeaderStatus.NO_HEADER
    assert texts(p) == [" | ".join(row)]


# ---------------------------------------------------------- label heuristic


@pytest.mark.parametrize(
    ("text", "label"),
    [
        ("Amount (LKR)", True),
        ("Approved budget", True),
        ("Date of completion", True),
        ("Staff (persons)", True),
        ("Cost (Rs. million)", True),
        ("Project", True),
        ("Project Alpha", False),
        ("pending", False),
        ("Project Alpha (Phase 2)", False),
        ("Budget 2025", True),
        ("FY 2024 budget", True),
        ("Phase 2", False),
        ("Ward 5", False),
        ("5,000,000", False),
        ("", False),
        ("Yes", False),
    ],
)
def test_is_label_like(text: str, label: bool) -> None:
    assert is_label_like(text) is label


# --------------------------------------------- adversarial: label-word data row


ADVERSARIAL = [["Project", "Amount"], ["Alpha", "5"]]


def test_adversarial_label_word_row_is_classified_as_header() -> None:
    """Documented, accepted ambiguity: content cannot tell a header made of generic
    labels from a data row that happens to contain them. The detector says HEADER;
    the decision and the withheld cells are recorded so it can be audited."""
    p = plan(ADVERSARIAL)
    assert p.tables_with_header == 1
    assert texts(p) == ["Project = Alpha | Amount = 5"]
    (decision,) = p.header_decisions
    assert decision["status"] == "header"
    assert "column labels" in str(decision["reason"])
    assert decision["header_span_ids"] == [f"{DOC}:#/tables/0:r0c0", f"{DOC}:#/tables/0:r0c1"]


def test_header_row_cells_are_preserved_not_discarded() -> None:
    """Even as a header, row 0 stays traceable: carried as header spans, shown to
    the model, listed in the decision, and never removed from the parsed spans."""
    source = cells(ADVERSARIAL)
    p = plan_sources(source, DocumentFormat.XLSX)
    (unit,) = p.batches[0]
    row0 = [c for c in source if c.location and c.location.table_row == 0]
    assert [h.span_id for h in unit.header_spans if h] == [c.span_id for c in row0]
    assert all(h is not None and h.location is not None for h in unit.header_spans)
    rendered = unit.render()
    assert "columns: Project | Amount" in rendered and "Project = Alpha" in rendered
    assert_no_cell_lost(ADVERSARIAL, p)
    assert len(source) == 4  # planning never mutates or drops the parsed spans


def test_never_policy_keeps_every_first_row_as_data() -> None:
    p = plan_sources(cells(ADVERSARIAL), DocumentFormat.XLSX, header_policy="never")
    assert texts(p) == ["Project | Amount", "Alpha | 5"]
    assert p.tables_with_header == 0
    (decision,) = p.header_decisions
    assert decision["status"] == "no_header" and "policy" in str(decision["reason"])
    assert all(h is None for b in p.batches for u in b for h in u.header_spans)


def test_never_policy_produces_no_header_evidence() -> None:
    """Without a header, 'Amount (LKR)' cannot ground the currency of the value."""
    rows = [["Project", "Amount (LKR)"], ["Project Alpha", "5500000"]]
    p = plan_sources(cells(rows), DocumentFormat.XLSX, header_policy="never")
    unit = p.batches[0][1]
    ec = ClaimValidator(p.batches[0], DOC, "RUN-t").validate_claim(
        LLMClaim.model_validate(
            llm_claim(
                source_id=unit.source_id,
                value="5500000",
                modality="unknown",
                evidence_quotes=["Project Alpha", "5500000"],
            )
        )
    )
    assert "currency_not_grounded" in ec.issues
    assert not [e for e in ec.claim.evidence if e.role == "header"]
    # The same table under the default policy keeps its header evidence.
    p2 = plan(rows)
    ec2 = ClaimValidator(p2.batches[0], DOC, "RUN-t").validate_claim(
        LLMClaim.model_validate(
            llm_claim(
                value="5500000", modality="unknown", evidence_quotes=["Project Alpha", "5500000"]
            )
        )
    )
    assert [(e.quote, e.role) for e in ec2.claim.evidence if e.role == "header"] == [
        ("LKR", "header")
    ]


def test_every_decision_is_recorded_with_a_reason() -> None:
    spans = cells([["Project", "Amount"], ["Alpha", "5"]])
    spans += [
        c.model_copy(
            update={
                "span_id": c.span_id.replace("#/tables/0", "#/tables/1"),
                "location": c.location.model_copy(update={"table_ref": "#/tables/1"}),
            }
        )  # type: ignore[union-attr]
        for c in cells([["Project Alpha", "pending"], ["Project Beta", "5"]])
    ]
    p = plan_sources(spans, DocumentFormat.XLSX)
    assert {(d["table_ref"], d["status"]) for d in p.header_decisions} == {
        ("#/tables/0", "header"),
        ("#/tables/1", "ambiguous"),
    }
    assert all(d["reason"] for d in p.header_decisions)
