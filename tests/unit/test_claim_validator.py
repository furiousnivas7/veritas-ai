"""Deterministic grounding of model candidates. No model involved: the 'model
output' is written by hand, so each rule is tested in isolation."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from factories import llm_claim, llm_entity, span, xlsx_table

from veritas.core.contracts import DocumentFormat, Modality, ObjectType, Quantity, SpanKind
from veritas.extraction.schema import LLMClaim, LLMEntity
from veritas.extraction.sources import SourceUnit, plan_sources
from veritas.extraction.validator import ClaimValidator, ExtractedClaim, Rejected

DOC = "DOC-000000000001"
BUDGET_2026 = (
    "Project Alpha has a proposed budget of LKR 7,000,000 for 2026. "
    "Project Alpha has an approved budget of LKR 5,000,000 for 2026. "
    "Project Alpha recorded actual expenditure of LKR 4,500,000 in 2026."
)


def units_for(*texts: str, heading: list[str] | None = None) -> list[SourceUnit]:
    spans = [
        span(DOC, f"#/texts/{i}", t, order=i, page=2, heading_path=heading)
        for i, t in enumerate(texts)
    ]
    (batch,) = plan_sources(spans, DocumentFormat.PDF).batches
    return batch


def validate(units: list[SourceUnit], **claim: Any) -> ExtractedClaim:
    return ClaimValidator(units, DOC, "RUN-test").validate_claim(
        LLMClaim.model_validate(llm_claim(**claim))
    )


def reject_reason(units: list[SourceUnit], **claim: Any) -> str:
    with pytest.raises(Rejected) as e:
        validate(units, **claim)
    return e.value.reason


# ----------------------------------------------------------------- simple facts


def test_simple_money_claim_with_provenance() -> None:
    units = units_for(
        "Project Alpha received an approved allocation of LKR 5 million in January 2025."
    )
    ec = validate(
        units, valid_from="2025-01", valid_to="2025-01", time_expression="in January 2025"
    )
    c = ec.claim
    assert (c.subject, c.predicate, c.modality) == ("Project Alpha", "budget", Modality.APPROVED)
    assert c.object == Quantity(value=Decimal("5000000"), currency="LKR")
    assert c.object_type is ObjectType.MONEY
    assert (c.valid_from, c.valid_to) == ("2025-01", "2025-01")
    text = units[0].spans[0].text
    # The model's quote, then the subject and time expression from the same sentence,
    # each added as its own exact evidence slice.
    assert [e.quote for e in c.evidence] == [
        "approved allocation of LKR 5 million",
        "Project Alpha",
        "in January 2025",
    ]
    for ev in c.evidence:
        assert ev.span_id == f"{DOC}:#/texts/0"
        assert ev.location.page_number == 2
        assert text[ev.char_start : ev.char_end] == ev.quote  # type: ignore[misc]
    assert ec.issues == []


def test_three_budget_statements_stay_distinct() -> None:
    """The critical semantic rule: proposed / approved / actual are not merged."""
    units = units_for(BUDGET_2026)
    claims = [
        validate(
            units,
            modality="proposed",
            value="7000000",
            predicate="proposed_budget",
            valid_from="2026",
            valid_to="2026",
            time_expression="for 2026",
            evidence_quotes=["proposed budget of LKR 7,000,000 for 2026"],
        ).claim,
        validate(
            units,
            modality="approved",
            value="5000000",
            predicate="budget",
            valid_from="2026",
            valid_to="2026",
            time_expression="for 2026",
            evidence_quotes=["approved budget of LKR 5,000,000 for 2026"],
        ).claim,
        validate(
            units,
            modality="actual",
            value="4500000",
            predicate="actual_expenditure",
            valid_from="2026",
            valid_to="2026",
            time_expression="in 2026",
            evidence_quotes=["recorded actual expenditure of LKR 4,500,000 in 2026"],
        ).claim,
    ]
    keys = {(c.predicate, c.modality, c.object.value) for c in claims}  # type: ignore[union-attr]
    assert keys == {
        ("budget", Modality.PROPOSED, Decimal("7000000")),
        ("budget", Modality.APPROVED, Decimal("5000000")),
        ("expenditure", Modality.ACTUAL, Decimal("4500000")),
    }
    assert all((c.valid_from, c.valid_to) == ("2026", "2026") for c in claims)
    assert all(isinstance(c.object, Quantity) and c.object.currency == "LKR" for c in claims)
    assert len({c.claim_id for c in claims}) == 3


def test_multiple_claims_in_one_element_get_separate_evidence() -> None:
    units = units_for(BUDGET_2026)
    a = validate(
        units,
        modality="proposed",
        value="7000000",
        evidence_quotes=["proposed budget of LKR 7,000,000"],
    ).claim
    b = validate(
        units,
        modality="approved",
        value="5000000",
        evidence_quotes=["approved budget of LKR 5,000,000"],
    ).claim
    assert a.evidence[0].span_id == b.evidence[0].span_id
    assert a.evidence[0].char_start != b.evidence[0].char_start


def test_quote_text_comes_from_source_not_model() -> None:
    units = units_for("Project Alpha received an APPROVED allocation of LKR 5\nmillion.")
    ev = validate(units, evidence_quotes=["approved allocation of lkr 5 million"]).claim.evidence[0]
    assert ev.quote == "APPROVED allocation of LKR 5\nmillion"


# ----------------------------------------------------------------- modality


def test_modality_contradicting_source_is_rejected() -> None:
    units = units_for("The project team proposes a budget of LKR 7 million for Project Alpha.")
    reason = reject_reason(
        units,
        modality="approved",
        value="7000000",
        evidence_quotes=["proposes a budget of LKR 7 million"],
    )
    assert reason == "modality_conflict"


def test_unstated_modality_is_downgraded_to_unknown() -> None:
    units = units_for("The total budget of Project Alpha is LKR 6 million.")
    ec = validate(
        units,
        modality="approved",
        value="6000000",
        evidence_quotes=["total budget of Project Alpha is LKR 6 million"],
    )
    assert ec.claim.modality is Modality.UNKNOWN
    assert "modality_not_grounded" in ec.issues
    assert ec.claim.extraction_confidence < 0.9


def test_modality_word_in_predicate_conflicting_with_field_is_rejected() -> None:
    units = units_for(BUDGET_2026)
    reason = reject_reason(
        units,
        predicate="approved_budget",
        modality="proposed",
        value="7000000",
        evidence_quotes=["proposed budget of LKR 7,000,000"],
    )
    assert reason == "predicate_modality_conflict"


def test_modality_from_predicate_when_field_unknown() -> None:
    units = units_for(BUDGET_2026)
    c = validate(
        units,
        predicate="approved_budget",
        modality="unknown",
        value="5000000",
        evidence_quotes=["approved budget of LKR 5,000,000"],
    ).claim
    assert (c.predicate, c.modality) == ("budget", Modality.APPROVED)


# ------------------------------------------------------- numbers and currency


def test_value_not_in_quote_is_rejected() -> None:
    units = units_for("Project Alpha received an approved allocation of LKR 5 million.")
    assert reject_reason(units, value="6000000") == "value_not_grounded"


def test_non_numeric_value_is_rejected() -> None:
    units = units_for("Project Alpha received an approved allocation of LKR 5 million.")
    assert reject_reason(units, value="five-ish") == "invalid_value"


def test_ungrounded_currency_is_removed_not_kept() -> None:
    units = units_for("Project Alpha received an approved allocation of 5 million.")
    ec = validate(units, currency="LKR", evidence_quotes=["approved allocation of 5 million"])
    assert ec.claim.object_type is ObjectType.NUMBER
    assert isinstance(ec.claim.object, Quantity) and ec.claim.object.currency is None
    assert "currency_not_grounded" in ec.issues


def test_percent() -> None:
    units = units_for("Project Alpha is 40% complete.")
    c = validate(
        units,
        predicate="completion",
        object_type="percent",
        value="40",
        currency=None,
        unit="percent",
        modality="unknown",
        evidence_quotes=["40% complete"],
    ).claim
    assert c.object == Quantity(value=Decimal("40"), unit="percent")


# ---------------------------------------------------------------- dates / time


def test_date_object() -> None:
    units = units_for("Project Alpha was completed in June 2025 under Dr. A. Perera.")
    c = validate(
        units,
        predicate="completion_date",
        object_type="date",
        value="2025-06",
        currency=None,
        modality="unknown",
        evidence_quotes=["completed in June 2025"],
    ).claim
    assert (c.object, c.object_type, c.predicate) == ("2025-06", ObjectType.DATE, "completion_date")


def test_date_object_not_in_quote_is_rejected() -> None:
    units = units_for("Project Alpha was completed in June 2025.")
    assert (
        reject_reason(
            units,
            predicate="completion_date",
            object_type="date",
            value="2025-09",
            currency=None,
            modality="unknown",
            evidence_quotes=["completed in June 2025"],
        )
        == "value_not_grounded"
    )


def test_ungrounded_period_is_removed() -> None:
    units = units_for("Project Alpha received an approved allocation of LKR 5 million.")
    ec = validate(units, valid_from="2026", valid_to="2026", time_expression="for 2026")
    assert (ec.claim.valid_from, ec.claim.valid_to, ec.claim.time_expression) == (None, None, None)
    assert "time_not_grounded" in ec.issues


def test_period_from_heading_context_is_not_evidence() -> None:
    """A heading such as 'Annual Report 2025' is not cited evidence and cannot supply
    a claim's period (changed 2026-11-28: previously kept with time_from_context)."""
    units = units_for(
        "Project Alpha received an approved allocation of LKR 5 million.",
        heading=["Annual Report 2025"],
    )
    ec = validate(units, valid_from="2025", valid_to="2025", time_expression="Annual Report 2025")
    assert (ec.claim.valid_from, ec.claim.valid_to, ec.claim.time_expression) == (None, None, None)
    assert "time_not_grounded" in ec.issues


# ------------------------------------------------------------ text / subject


def test_text_object_and_negation() -> None:
    units = units_for("Project Alpha was not completed in 2025.")
    c = validate(
        units,
        predicate="status",
        object_type="text",
        value="completed",
        currency=None,
        modality="unknown",
        negated=True,
        evidence_quotes=["Project Alpha was not completed"],
    ).claim
    assert c.negated and c.object == "completed"


def test_subject_must_appear_in_source() -> None:
    units = units_for("Project Alpha received an approved allocation of LKR 5 million.")
    assert reject_reason(units, subject="Project Gamma") == "subject_not_grounded"


# ---------------------------------------------------------- references


def test_unknown_source_id_is_rejected() -> None:
    units = units_for("Project Alpha received an approved allocation of LKR 5 million.")
    assert reject_reason(units, source_id="S9") == "unknown_source"


def test_invented_quote_is_rejected() -> None:
    units = units_for("Project Alpha received an approved allocation of LKR 5 million.")
    assert (
        reject_reason(units, evidence_quotes=["approved budget of LKR 5 million"])
        == "quote_not_in_source"
    )


def test_one_bad_quote_rejects_the_claim() -> None:
    units = units_for("Project Alpha received an approved allocation of LKR 5 million.")
    reason = reject_reason(
        units, evidence_quotes=["approved allocation of LKR 5 million", "on page 12 of the report"]
    )
    assert reason == "quote_not_in_source"


def test_unlocated_span_cannot_be_evidence() -> None:
    """Missing provenance: even if the model quotes it correctly, it cannot be cited."""
    s = span(
        DOC,
        "#/texts/0",
        "Project Alpha received an approved allocation of LKR 5 million.",
        located=False,
    )
    unit = SourceUnit("S1", "text", s.text, "", (s,))
    assert reject_reason([unit]) == "missing_provenance"


def test_planner_excludes_unlocated_spans() -> None:
    spans = [
        span(DOC, "#/texts/0", "located text", order=0),
        span(DOC, "#/texts/1", "unlocated text", order=1, located=False),
    ]
    plan = plan_sources(spans, DocumentFormat.PDF)
    assert plan.skipped_missing_provenance == 1
    assert [u.text for b in plan.batches for u in b] == ["located text"]


# --------------------------------------------------------------- XLSX / DOCX


def test_xlsx_row_claim_cites_cells() -> None:
    cells = xlsx_table(
        DOC,
        "Budget",
        [
            ["Project", "Modality", "Amount (LKR)", "Date"],
            ["Project Alpha", "approved", "5500000", "2025-01"],
        ],
    )
    (units,) = plan_sources(cells, DocumentFormat.XLSX).batches
    (row,) = units
    assert row.kind == "table_row"
    assert "Amount (LKR) = 5500000" in row.text
    c = validate(
        units,
        value="5500000",
        valid_from="2025-01",
        valid_to="2025-01",
        time_expression="2025-01",
        evidence_quotes=["Project Alpha", "approved", "5500000", "2025-01"],
    ).claim
    data = {
        (e.location.sheet_name, e.location.cell_range) for e in c.evidence if e.role != "header"
    }
    assert data == {("Budget", "A2"), ("Budget", "B2"), ("Budget", "C2"), ("Budget", "D2")}
    # The currency comes from the value column's header, recorded separately.
    assert [(e.location.cell_range, e.quote) for e in c.evidence if e.role == "header"] == [
        ("C1", "LKR")
    ]
    assert c.modality is Modality.APPROVED and c.object.currency == "LKR"  # type: ignore[union-attr]


def test_xlsx_header_cells_are_context_not_evidence() -> None:
    cells = xlsx_table(DOC, "Budget", [["Project", "Amount (LKR)"], ["Project Alpha", "5500000"]])
    (units,) = plan_sources(cells, DocumentFormat.XLSX).batches
    assert (
        reject_reason(units, modality="unknown", evidence_quotes=["Amount (LKR)"], value="5500000")
        == "quote_not_in_source"
    )


def test_docx_claim_keeps_structural_location() -> None:
    s = span(
        DOC,
        "#/texts/2",
        "The project team proposes a budget of LKR 7 million for Project Alpha.",
        fmt=DocumentFormat.DOCX,
        page=None,
        heading_path=["Funding Proposal", "Budget request"],
    )
    (units,) = plan_sources([s], DocumentFormat.DOCX).batches
    c = validate(
        units,
        modality="proposed",
        value="7000000",
        evidence_quotes=["proposes a budget of LKR 7 million"],
    ).claim
    loc = c.evidence[0].location
    assert loc.page_number is None and loc.element_id == "#/texts/2"
    assert loc.heading_path == ["Funding Proposal", "Budget request"]


# ------------------------------------------------------------------- entities


def test_entity_mention_offsets_and_aliases() -> None:
    text = "Research Office of the University of Sri Jayewardenepura (USJ) during 2025."
    units = units_for(text)
    v = ClaimValidator(units, DOC, "RUN-test")
    mentions, missing = v.validate_entity(
        LLMEntity.model_validate(
            llm_entity(
                name="University of Sri Jayewardenepura",
                entity_type="organization",
                aliases=["USJ", "Sri Jayewardenepura Uni"],
            )
        )
    )
    assert [m.surface for m in mentions] == ["University of Sri Jayewardenepura", "USJ"]
    assert all(text[m.char_start : m.char_end] == m.surface for m in mentions)
    assert missing == ["Sri Jayewardenepura Uni"]


def test_entity_not_in_source_is_rejected() -> None:
    v = ClaimValidator(units_for("Project Alpha was completed."), DOC, "RUN-test")
    with pytest.raises(Rejected) as e:
        v.validate_entity(LLMEntity.model_validate(llm_entity(name="Project Beta")))
    assert e.value.reason == "entity_not_in_source"


# --------------------------------------------------------------- determinism


def test_validation_is_deterministic_and_model_independent() -> None:
    units = units_for(BUDGET_2026)
    kw = dict(
        modality="approved", value="5000000", evidence_quotes=["approved budget of LKR 5,000,000"]
    )
    a = ClaimValidator(units, DOC, "RUN-a").validate_claim(LLMClaim.model_validate(llm_claim(**kw)))
    b = ClaimValidator(units, DOC, "RUN-b").validate_claim(LLMClaim.model_validate(llm_claim(**kw)))
    assert a.fingerprint == b.fingerprint
    strip = {
        "claim_id": True,
        "evidence": {"__all__": {"evidence_id"}},
        "metadata": {"__all__": {"evidence_ids"}},  # per-claim ids, like evidence_id
    }
    assert a.claim.model_dump(exclude=strip) == b.claim.model_dump(exclude=strip)


def test_headings_are_context_not_sources() -> None:
    spans = [
        span(DOC, "#/texts/0", "2. Project Alpha", kind=SpanKind.HEADING, order=0),
        span(DOC, "#/texts/1", "It was completed.", order=1, heading_path=["2. Project Alpha"]),
    ]
    (batch,) = plan_sources(spans, DocumentFormat.PDF).batches
    assert [u.text for u in batch] == ["It was completed."]
    assert batch[0].context == "2. Project Alpha"


# ------------------------------------------- regressions from real model runs


def _budget_rows() -> list[SourceUnit]:
    cells = xlsx_table(
        DOC,
        "Budget",
        [
            ["Project", "Modality", "Amount (LKR)", "Date"],
            ["Project Alpha", "approved", "5500000", "2025-01"],
        ],
    )
    (units,) = plan_sources(cells, DocumentFormat.XLSX).batches
    return units


def test_row_rendering_quote_is_split_into_cells() -> None:
    """qwen3:4b quoted 'Header = value' pairs exactly as the row was rendered."""
    c = validate(
        _budget_rows(),
        value="5500000",
        evidence_quotes=["Project = Project Alpha | Amount (LKR) = 5500000"],
    ).claim
    assert [(e.location.cell_range, e.quote, e.role) for e in c.evidence] == [
        ("A2", "Project Alpha", "quote"),
        ("C2", "5500000", "quote"),
        ("C1", "LKR", "header"),  # header text quoted by the model is not data evidence
        # enrichment: the row's labelled Date cell is the claim's period
        ("D2", "2025-01", "scope"),
        ("D1", "Date", "header"),
    ]


def test_row_pair_with_wrong_header_is_rejected() -> None:
    """The value must sit under the quoted header; a matching value elsewhere is not enough."""
    assert (
        reject_reason(_budget_rows(), value="5500000", evidence_quotes=["Date = 5500000"])
        == "quote_not_in_source"
    )


def test_row_pair_with_unknown_header_is_rejected() -> None:
    assert (
        reject_reason(
            _budget_rows(), value="5500000", evidence_quotes=["Approved budget (LKR) = 5500000"]
        )
        == "quote_not_in_source"
    )


def test_heading_words_do_not_ground_modality() -> None:
    """qwen3:4b labelled 'X will host the project' as proposed, under a heading
    'Funding Proposal'. Headings are not modality evidence."""
    s = span(
        DOC,
        "#/texts/3",
        "Sri Jayewardenepura University will host the project.",
        fmt=DocumentFormat.DOCX,
        page=None,
        heading_path=["Project Alpha — Funding Proposal", "Budget request"],
    )
    (units,) = plan_sources([s], DocumentFormat.DOCX).batches
    ec = validate(
        units,
        subject="Sri Jayewardenepura University",
        predicate="host_institution",
        object_type="text",
        value="the project",
        currency=None,
        modality="proposed",
        evidence_quotes=["Sri Jayewardenepura University will host the project."],
    )
    assert ec.claim.modality is Modality.UNKNOWN
    assert "modality_not_grounded" in ec.issues


# ---------------------------------------------------- currency must be quoted


def test_currency_cannot_be_borrowed_from_elsewhere_in_the_source() -> None:
    units = units_for("The LKR budget table follows. Project Alpha hired 300 staff in 2025.")
    ec = validate(
        units,
        predicate="staff_count",
        value="300",
        currency="LKR",
        modality="unknown",
        evidence_quotes=["Project Alpha hired 300 staff"],
    )
    assert ec.claim.object_type is ObjectType.NUMBER
    assert isinstance(ec.claim.object, Quantity) and ec.claim.object.currency is None
    assert "currency_not_grounded" in ec.issues


def test_currency_in_heading_context_is_not_evidence() -> None:
    units = units_for(
        "Project Alpha received an approved allocation of 5 million.", heading=["Budget (LKR)"]
    )
    ec = validate(units, evidence_quotes=["approved allocation of 5 million"])
    assert "currency_not_grounded" in ec.issues


@pytest.mark.parametrize(
    ("text", "quote", "code"),
    [
        ("an approved allocation of LKR 5 million", "approved allocation of LKR 5 million", "LKR"),
        ("an approved allocation of USD 5 million", "approved allocation of USD 5 million", "USD"),
        ("an approved allocation of US$ 5 million", "approved allocation of US$ 5 million", "USD"),
        ("an approved allocation of EUR 5 million", "approved allocation of EUR 5 million", "EUR"),
    ],
)
def test_currency_in_quote_is_accepted(text: str, quote: str, code: str) -> None:
    units = units_for(f"Project Alpha received {text}.")
    c = validate(units, currency=code, evidence_quotes=[quote]).claim
    assert c.object_type is ObjectType.MONEY
    assert isinstance(c.object, Quantity) and c.object.currency == code


def test_currency_from_cited_cell_header_is_accepted() -> None:
    c = validate(_budget_rows(), value="5500000", evidence_quotes=["5500000"]).claim
    assert c.object.currency == "LKR"  # type: ignore[union-attr]  # header "Amount (LKR)"


def test_currency_from_another_column_header_is_not_borrowed() -> None:
    cells = xlsx_table(
        DOC, "Budget", [["Project", "Staff", "Notes (LKR)"], ["Project Alpha", "42", "pending"]]
    )
    (units,) = plan_sources(cells, DocumentFormat.XLSX).batches
    ec = validate(
        units,
        predicate="staff_count",
        value="42",
        currency="LKR",
        modality="unknown",
        evidence_quotes=["Project Alpha", "42"],
    )
    assert "currency_not_grounded" in ec.issues


# ------------------------------------------------------- booleans are grounded


def _bool(units: list[SourceUnit], value: str, quote: str, **kw: Any) -> ExtractedClaim:
    return validate(
        units,
        predicate=kw.pop("predicate", "completed"),
        object_type="boolean",
        value=value,
        currency=None,
        modality="unknown",
        evidence_quotes=[quote],
        **kw,
    )


def test_boolean_positive_statement() -> None:
    units = units_for("Project Alpha was completed in June 2025.")
    assert _bool(units, "true", "Project Alpha was completed").claim.object is True


def test_boolean_negative_statement() -> None:
    units = units_for("Project Alpha was not completed in 2025.")
    assert _bool(units, "false", "Project Alpha was not completed").claim.object is False
    # Equivalent encoding: value true + negated
    c = _bool(units, "true", "Project Alpha was not completed", negated=True).claim
    assert c.object is True and c.negated


def test_boolean_contradicting_source_is_rejected() -> None:
    units = units_for("Project Alpha was not completed in 2025.")
    with pytest.raises(Rejected) as e:
        _bool(units, "true", "Project Alpha was not completed")
    assert e.value.reason == "boolean_conflict"


def test_boolean_hedged_statement_is_rejected() -> None:
    units = units_for("Project Alpha may be completed in 2026.")
    with pytest.raises(Rejected) as e:
        _bool(units, "true", "Project Alpha may be completed")
    assert e.value.reason == "boolean_not_grounded"


def test_boolean_about_another_subject_is_rejected() -> None:
    units = units_for("Project Alpha started in 2024. Project Beta was completed in 2025.")
    with pytest.raises(Rejected) as e:
        _bool(units, "true", "Project Beta was completed")
    # Rejected before polarity is even checked: Project Alpha is not in that sentence.
    assert e.value.reason == "subject_not_grounded"


@pytest.mark.parametrize(
    "text",
    [
        "Project Beta was completed, unlike Project Alpha.",
        "Unlike Project Alpha, Project Beta was completed.",
        "Project Beta was completed before work on Project Alpha began.",
    ],
)
def test_boolean_predicate_of_another_subject_in_same_sentence_is_rejected(text: str) -> None:
    units = units_for(text)
    with pytest.raises(Rejected) as e:
        _bool(units, "true", "Project Beta was completed")
    assert e.value.reason == "boolean_not_grounded"


def test_boolean_unrelated_predicate_is_rejected() -> None:
    units = units_for("Project Alpha was approved in 2025.")
    with pytest.raises(Rejected) as e:
        _bool(units, "true", "Project Alpha was approved")
    assert e.value.reason == "boolean_not_grounded"


def test_boolean_from_table_cell() -> None:
    cells = xlsx_table(DOC, "Status", [["Project", "Completed"], ["Project Alpha", "No"]])
    (units,) = plan_sources(cells, DocumentFormat.XLSX).batches
    c = validate(
        units,
        predicate="completed",
        object_type="boolean",
        value="false",
        currency=None,
        modality="unknown",
        evidence_quotes=["Project Alpha", "No"],
    ).claim
    assert c.object is False
    with pytest.raises(Rejected) as e:
        validate(
            units,
            predicate="completed",
            object_type="boolean",
            value="true",
            currency=None,
            modality="unknown",
            evidence_quotes=["Project Alpha", "No"],
        )
    assert e.value.reason == "boolean_conflict"


# --------------------------------------------------- repeated quote offsets


def test_repeated_quote_keeps_selected_occurrence_offsets() -> None:
    text = "LKR 5 million was proposed. Later LKR 5 million was approved."
    s = span(DOC, "#/texts/0", text)
    v = ClaimValidator([SourceUnit("S1", "text", text, "", (s,))], DOC, "RUN-test")
    second = text.index("LKR 5 million", 5)
    ev = v._make_evidence(s, second, second + len("LKR 5 million"))
    assert (ev.char_start, ev.char_end) == (second, second + 13)
    assert text[ev.char_start : ev.char_end] == ev.quote  # type: ignore[misc]
    first = v._make_evidence(s, 0, 13)
    assert first.char_start == 0 and first.evidence_id != ev.evidence_id


def test_exact_match_is_preferred_over_an_earlier_whitespace_variant() -> None:
    """locate() prefers an exact occurrence; offsets must be that occurrence's."""
    text = "LKR 5\nmillion was proposed. Project Alpha then had LKR 5 million approved."
    units = units_for(text)
    ev = validate(
        units, value="5000000", modality="unknown", evidence_quotes=["LKR 5 million"]
    ).claim.evidence[0]
    assert ev.char_start == text.index("LKR 5 million")
    assert text[ev.char_start : ev.char_end] == "LKR 5 million"  # type: ignore[misc]


# =====================================================================
# Grounding scope corrections (2026-11-28): everything a claim relies on
# must be in its cited evidence: a quote, the quote's own sentence, or the
# cited cell's own table row. Never elsewhere in the unit.
# =====================================================================


def _status_cells(rows: list[list[str]]) -> list[SourceUnit]:
    return plan_sources(
        xlsx_table(DOC, "Status", [["Project", "Completed"], *rows]), DocumentFormat.XLSX
    ).batches[0]


def _multi_row_unit(rows: list[list[str]]) -> SourceUnit:
    """One unit holding several rows (the planner makes one unit per row; this
    checks that the validator itself never mixes rows)."""
    cells = [
        c
        for c in xlsx_table(DOC, "Status", [["Project", "Completed"], *rows])
        if c.location and c.location.table_row
    ]
    labels = tuple("Project" if c.location.table_col == 0 else "Completed" for c in cells)  # type: ignore[union-attr]
    return SourceUnit(
        "S1", "table_row", " / ".join(c.text for c in cells), "", tuple(cells), labels
    )


def _bool_row(
    unit: SourceUnit | list[SourceUnit],
    subject: str,
    value: str,
    quotes: list[str],
    source_id: str = "S1",
) -> ExtractedClaim:
    units = unit if isinstance(unit, list) else [unit]
    return validate(
        units,
        source_id=source_id,
        subject=subject,
        predicate="completed",
        object_type="boolean",
        value=value,
        currency=None,
        modality="unknown",
        evidence_quotes=quotes,
    )


def _reason(fn: Any) -> str:
    with pytest.raises(Rejected) as e:
        fn()
    return e.value.reason


# ------------------------------------------------- boolean, row by row


def test_boolean_rows_via_planner_alpha_false_beta_true() -> None:
    units = _status_cells([["Project Alpha", "No"], ["Project Beta", "Yes"]])
    assert _bool_row(units, "Project Alpha", "false", ["Project Alpha", "No"]).claim.object is False
    assert (
        _bool_row(
            units, "Project Beta", "true", ["Project Beta", "Yes"], source_id="S2"
        ).claim.object
        is True
    )
    assert (
        _reason(lambda: _bool_row(units, "Project Alpha", "true", ["Project Alpha", "No"]))
        == "boolean_conflict"
    )


def test_boolean_rows_in_one_unit_are_judged_independently() -> None:
    unit = _multi_row_unit([["Project Alpha", "No"], ["Project Beta", "Yes"]])
    assert _bool_row(unit, "Project Alpha", "false", ["Project Alpha", "No"]).claim.object is False
    assert _bool_row(unit, "Project Beta", "true", ["Project Beta", "Yes"]).claim.object is True


def test_positive_value_in_another_entitys_row_never_validates_subject() -> None:
    unit = _multi_row_unit([["Project Alpha", "No"], ["Project Beta", "Yes"]])
    # Alpha quoted from its row, but the "Yes" cited belongs to Beta's row.
    assert (
        _reason(lambda: _bool_row(unit, "Project Alpha", "true", ["Project Alpha", "Yes"]))
        == "boolean_not_grounded"
    )
    # Only Beta's row cited: Alpha is not in the cited evidence at all.
    assert (
        _reason(lambda: _bool_row(unit, "Project Alpha", "true", ["Project Beta", "Yes"]))
        == "subject_not_grounded"
    )


def test_conflicting_matching_rows_are_ambiguous() -> None:
    unit = _multi_row_unit([["Project Alpha", "No"], ["Project Alpha", "Yes"]])
    assert (
        _reason(lambda: _bool_row(unit, "Project Alpha", "true", ["Project Alpha", "No", "Yes"]))
        == "boolean_not_grounded"
    )
    assert (
        _reason(lambda: _bool_row(unit, "Project Alpha", "false", ["Project Alpha", "No", "Yes"]))
        == "boolean_not_grounded"
    )


# ------------------------------------------------- subject and text value


def test_subject_in_an_unrelated_sentence_is_rejected() -> None:
    units = units_for("Project Alpha started in 2024. The budget of LKR 5 million was approved.")
    assert (
        reject_reason(units, evidence_quotes=["budget of LKR 5 million was approved"])
        == "subject_not_grounded"
    )


def test_subject_in_heading_only_is_rejected() -> None:
    units = units_for(
        "It received an approved allocation of LKR 5 million.", heading=["2. Project Alpha"]
    )
    assert reject_reason(units) == "subject_not_grounded"


def test_subject_in_quote_sentence_is_accepted_and_cited() -> None:
    text = "Project Alpha received an approved allocation of LKR 5 million."
    c = validate(units_for(text)).claim
    subj = next(e for e in c.evidence if e.quote == "Project Alpha")
    assert (subj.char_start, subj.char_end) == (0, 13)


def test_text_value_in_an_unrelated_sentence_is_rejected() -> None:
    units = units_for("Dr. A. Perera supervises Project Beta. Project Alpha has a new supervisor.")
    assert (
        reject_reason(
            units,
            predicate="supervisor",
            object_type="text",
            value="Dr. A. Perera",
            currency=None,
            modality="unknown",
            evidence_quotes=["Project Alpha has a new supervisor"],
        )
        == "value_not_grounded"
    )


def test_text_value_in_quote_sentence_is_accepted_and_cited() -> None:
    text = "Project Alpha was completed in June 2025 under the supervision of Dr. A. Perera."
    c = validate(
        units_for(text),
        predicate="supervisor",
        object_type="text",
        value="Dr. A. Perera",
        currency=None,
        modality="unknown",
        evidence_quotes=["Project Alpha was completed"],
    ).claim
    assert c.object == "Dr. A. Perera"
    assert "Dr. A. Perera" in [e.quote for e in c.evidence]


def test_table_subject_from_same_row_is_accepted_and_cited() -> None:
    """Row-level context stays valid: the subject cell joins the evidence."""
    c = validate(_budget_rows(), value="5500000", evidence_quotes=["5500000"]).claim
    assert [(e.location.cell_range, e.quote, e.role) for e in c.evidence] == [
        ("C2", "5500000", "quote"),
        ("A2", "Project Alpha", "scope"),
        ("C1", "LKR", "header"),
        ("D2", "2025-01", "scope"),  # enrichment: period from the row's Date cell
        ("D1", "Date", "header"),
    ]


# ------------------------------------------------------------- units


def _number(units: list[SourceUnit], value: str, unit: str, quotes: list[str]) -> ExtractedClaim:
    return validate(
        units,
        predicate="staff_count",
        object_type="number",
        value=value,
        unit=unit,
        currency=None,
        modality="unknown",
        evidence_quotes=quotes,
    )


def test_unit_in_evidence_quote_is_kept() -> None:
    c = _number(
        units_for("Project Alpha hired 300 staff."),
        "300",
        "staff",
        ["Project Alpha hired 300 staff"],
    ).claim
    assert c.object == Quantity(value=Decimal("300"), unit="staff")


def test_unit_in_cited_cell_header_is_kept() -> None:
    cells = xlsx_table(DOC, "Teams", [["Project", "Staff (persons)"], ["Project Alpha", "42"]])
    (units,) = plan_sources(cells, DocumentFormat.XLSX).batches
    c = _number(units, "42", "persons", ["Project Alpha", "42"]).claim
    assert c.object == Quantity(value=Decimal("42"), unit="persons")


def test_unit_only_in_unrelated_sentence_is_dropped() -> None:
    ec = _number(
        units_for("Project Alpha hired 300 people. Each team has 5 staff."),
        "300",
        "staff",
        ["Project Alpha hired 300 people"],
    )
    assert ec.claim.object == Quantity(value=Decimal("300"), unit=None)
    assert "unit_not_grounded" in ec.issues


def test_unit_in_another_column_header_is_dropped() -> None:
    cells = xlsx_table(
        DOC, "Sites", [["Project", "Count", "Area (sqm)"], ["Project Alpha", "42", "100"]]
    )
    (units,) = plan_sources(cells, DocumentFormat.XLSX).batches
    ec = _number(units, "42", "sqm", ["Project Alpha", "42", "100"])
    assert ec.claim.object == Quantity(value=Decimal("42"), unit=None)
    assert "unit_not_grounded" in ec.issues


def test_currency_from_another_cited_column_header_is_not_borrowed() -> None:
    cells = xlsx_table(
        DOC, "Budget", [["Project", "Staff", "Notes (LKR)"], ["Project Alpha", "42", "pending"]]
    )
    (units,) = plan_sources(cells, DocumentFormat.XLSX).batches
    ec = validate(
        units,
        predicate="staff_count",
        value="42",
        currency="LKR",
        modality="unknown",
        evidence_quotes=["Project Alpha", "42", "pending"],
    )
    assert "currency_not_grounded" in ec.issues


# --------------------------------------------------------------- time


TWO_PROJECTS = (
    "Project Alpha received LKR 5 million in 2025. Project Beta received LKR 3 million in 2026."
)


def test_time_cannot_be_borrowed_from_another_projects_sentence() -> None:
    ec = validate(
        units_for(TWO_PROJECTS),
        modality="unknown",
        evidence_quotes=["Project Alpha received LKR 5 million"],
        valid_from="2026",
        valid_to="2026",
        time_expression="in 2026",
    )
    assert (ec.claim.valid_from, ec.claim.time_expression) == (None, None)
    assert "time_not_grounded" in ec.issues


def test_time_from_the_claims_own_sentence_is_kept_and_cited() -> None:
    ec = validate(
        units_for(TWO_PROJECTS),
        modality="unknown",
        evidence_quotes=["Project Alpha received LKR 5 million"],
        valid_from="2025",
        valid_to="2025",
        time_expression="in 2025",
    )
    assert ec.claim.valid_from == "2025"
    t = next(e for e in ec.claim.evidence if e.quote == "in 2025")
    assert TWO_PROJECTS[t.char_start : t.char_end] == "in 2025"  # type: ignore[misc]
    assert t.char_start < TWO_PROJECTS.index("Project Beta")  # type: ignore[operator]


def test_time_expression_must_state_the_dates() -> None:
    ec = validate(
        units_for(TWO_PROJECTS),
        modality="unknown",
        evidence_quotes=["Project Alpha received LKR 5 million"],
        valid_from="2026",
        valid_to="2026",
        time_expression="in 2025",
    )
    assert "time_not_grounded" in ec.issues


def test_time_from_same_table_row_is_kept_and_cited() -> None:
    c = validate(
        _budget_rows(),
        value="5500000",
        evidence_quotes=["Project Alpha", "5500000"],
        valid_from="2025-01",
        valid_to="2025-01",
        time_expression="2025-01",
    ).claim
    assert c.valid_from == "2025-01"
    assert ("D2", "2025-01") in [(e.location.cell_range, e.quote) for e in c.evidence]


def test_time_from_sheet_name_is_not_evidence() -> None:
    cells = xlsx_table(
        DOC, "Budget 2025", [["Project", "Amount (LKR)"], ["Project Alpha", "5500000"]]
    )
    (units,) = plan_sources(cells, DocumentFormat.XLSX).batches
    ec = validate(
        units,
        value="5500000",
        modality="unknown",
        evidence_quotes=["Project Alpha", "5500000"],
        valid_from="2025",
        valid_to="2025",
        time_expression="Budget 2025",
    )
    assert "time_not_grounded" in ec.issues


# ----------------------------------------------------------- modality scope


def test_modality_word_in_another_sentence_does_not_ground() -> None:
    units = units_for(
        "The plan was approved last year. Project Alpha has a budget of LKR 6 million."
    )
    ec = validate(units, value="6000000", evidence_quotes=["budget of LKR 6 million"])
    assert ec.claim.modality is Modality.UNKNOWN
    assert "modality_not_grounded" in ec.issues


def test_modality_word_elsewhere_in_quote_sentence_still_grounds() -> None:
    units = units_for("Project Alpha has an approved budget of LKR 6 million.")
    ec = validate(units, value="6000000", evidence_quotes=["budget of LKR 6 million"])
    assert ec.claim.modality is Modality.APPROVED


def test_day_of_a_date_cannot_ground_a_quantity() -> None:
    units = units_for("Project Alpha was completed on 2025-01-15.")
    assert (
        reject_reason(
            units,
            predicate="staff_count",
            object_type="number",
            value="15",
            currency=None,
            modality="unknown",
            evidence_quotes=["Project Alpha was completed on 2025-01-15"],
        )
        == "value_not_grounded"
    )


# ------------------------------------------------------------ table headers


def _units(rows: list[list[str]], sheet: str = "S") -> list[SourceUnit]:
    return plan_sources(xlsx_table(DOC, sheet, rows), DocumentFormat.XLSX).batches[0]


def test_headered_table_uses_row_zero_as_header() -> None:
    units = _units(
        [["Project", "Amount (LKR)"], ["Project Alpha", "5500000"], ["Project Beta", "2500000"]]
    )
    assert [u.text for u in units] == [
        "Project = Project Alpha | Amount (LKR) = 5500000",
        "Project = Project Beta | Amount (LKR) = 2500000",
    ]
    assert [h.text if h else None for h in units[0].header_spans] == ["Project", "Amount (LKR)"]


def test_headerless_table_keeps_first_data_row() -> None:
    units = _units(
        [["Project Alpha", "5500000", "2025-01"], ["Project Beta", "2500000", "2025-03"]]
    )
    assert [u.text for u in units] == [
        "Project Alpha | 5500000 | 2025-01",
        "Project Beta | 2500000 | 2025-03",
    ]
    assert all(h is None for u in units for h in u.header_spans)


def test_all_text_table_is_treated_as_headerless() -> None:
    """Documented limitation: without a value-like column the header cannot be told
    apart from data, so nothing is discarded."""
    units = _units([["Project", "Status"], ["Project Alpha", "Active"]])
    assert [u.text for u in units] == ["Project | Status", "Project Alpha | Active"]


def test_yes_no_column_identifies_header() -> None:
    units = _units([["Project", "Completed"], ["Project Alpha", "No"]])
    assert [u.text for u in units] == ["Project = Project Alpha | Completed = No"]


def test_irregular_rows_keep_their_columns() -> None:
    # header lacks a label for column 2; second data row is short; third is wider
    cells = [
        c
        for c in xlsx_table(
            DOC,
            "S",
            [
                ["Project", "Amount", "x"],
                ["Project Alpha", "5500000", "2025-01"],
                ["Project Beta", "2500000", "y"],
                ["Project Gamma", "100", "z"],
            ],
        )
        if not (c.location and c.location.table_row == 0 and c.location.table_col == 2)
        and not (c.location and c.location.table_row == 2 and c.location.table_col == 2)
    ]
    units = plan_sources(cells, DocumentFormat.XLSX).batches[0]
    assert [u.text for u in units] == [
        "Project = Project Alpha | Amount = 5500000 | 2025-01",
        "Project = Project Beta | Amount = 2500000",
        "Project = Project Gamma | Amount = 100 | z",
    ]


def test_value_like_first_row_is_never_a_header() -> None:
    units = _units([["Project Alpha", "approved", "5500000"], ["Project Beta", "actual", "2"]])
    assert len(units) == 2 and units[0].text.startswith("Project Alpha")


def test_single_row_table_has_no_header() -> None:
    units = _units([["Project", "Amount"]])
    assert [u.text for u in units] == ["Project | Amount"]


# ------------------------------------------------------ header provenance


def test_value_cell_and_supporting_header_are_both_traceable() -> None:
    units = _budget_rows()
    c = validate(units, value="5500000", evidence_quotes=["Project Alpha", "5500000"]).claim
    value = next(e for e in c.evidence if e.quote == "5500000")
    header = next(e for e in c.evidence if e.role == "header")
    assert (value.role, value.location.cell_range) == ("quote", "C2")
    assert (header.quote, header.location.cell_range, header.location.sheet_name) == (
        "LKR",
        "C1",
        "Budget",
    )
    header_span = units[0].header_spans[2]
    assert header_span is not None and header.span_id == header_span.span_id
    assert header_span.text[header.char_start : header.char_end] == "LKR"  # type: ignore[misc]
    assert header.span_id != value.span_id


def test_unit_from_header_is_recorded_as_header_evidence() -> None:
    cells = xlsx_table(DOC, "Teams", [["Project", "Staff (persons)"], ["Project Alpha", "42"]])
    (units,) = plan_sources(cells, DocumentFormat.XLSX).batches
    c = _number(units, "42", "persons", ["Project Alpha", "42"]).claim
    (h,) = [e for e in c.evidence if e.role == "header"]
    assert (h.quote, h.location.cell_range) == ("persons", "B1")


def test_boolean_header_is_recorded_as_header_evidence() -> None:
    units = _status_cells([["Project Alpha", "No"]])
    c = _bool_row(units, "Project Alpha", "false", ["Project Alpha", "No"]).claim
    assert [(e.quote, e.location.cell_range) for e in c.evidence if e.role == "header"] == [
        ("Completed", "B1")
    ]


def test_unlocated_header_cell_is_not_traceable_support() -> None:
    """A header without provenance is excluded by the planner, so it cannot ground."""
    cells = xlsx_table(DOC, "Budget", [["Project", "Amount (LKR)"], ["Project Alpha", "5500000"]])
    cells = [
        span(
            DOC,
            c.span_id.split(":", 1)[1].split(":r")[0],
            c.text,
            fmt=DocumentFormat.XLSX,
            kind=SpanKind.TABLE_CELL,
            page=None,
            located=False,
            order=c.order,
        )
        if c.location and c.location.table_row == 0 and c.location.table_col == 1
        else c
        for c in cells
    ]
    (units,) = plan_sources(cells, DocumentFormat.XLSX).batches
    # Without its located "Amount (LKR)" cell, row 0 no longer qualifies as a header,
    # so it is kept as a data row and Project Alpha's row is the second unit.
    alpha = next(u for u in units if "Project Alpha" in u.text)
    assert all(h is None for h in alpha.header_spans)
    ec = validate(
        units,
        source_id=alpha.source_id,
        value="5500000",
        evidence_quotes=["Project Alpha", "5500000"],
    )
    assert "currency_not_grounded" in ec.issues
    assert not [e for e in ec.claim.evidence if e.role == "header"]


def test_header_span_without_location_is_rejected_as_support() -> None:
    """Even if a unit is built by hand with an unlocated header, it is not evidence."""
    (unit,) = _budget_rows()
    bad = tuple(
        span(
            DOC,
            "#/tables/0",
            h.text,
            fmt=DocumentFormat.XLSX,
            kind=SpanKind.TABLE_CELL,
            page=None,
            located=False,
        )
        if h is not None
        else None
        for h in unit.header_spans
    )
    hand_built = SourceUnit(
        unit.source_id, unit.kind, unit.text, unit.context, unit.spans, unit.labels, bad
    )
    ec = validate([hand_built], value="5500000", evidence_quotes=["Project Alpha", "5500000"])
    assert "currency_not_grounded" in ec.issues


# ----------------------------------------------------------- modality scope


def test_modality_single_in_sentence_is_accepted() -> None:
    units = units_for("Project Alpha has an approved budget of LKR 6 million.")
    assert (
        validate(units, value="6000000", evidence_quotes=["budget of LKR 6 million"]).claim.modality
        is Modality.APPROVED
    )


def test_conflicting_modalities_in_one_sentence_are_ambiguous() -> None:
    units = units_for("Project Alpha was proposed, but later approved, with LKR 6 million.")
    for model in ("proposed", "approved"):
        ec = validate(
            units,
            modality=model,
            value="6000000",
            evidence_quotes=["Project Alpha was proposed", "LKR 6 million"]
            if model == "proposed"
            else ["later approved, with LKR 6 million"],
        )
        assert ec.claim.modality is Modality.UNKNOWN, model
        assert "modality_ambiguous" in ec.issues


def test_quoting_only_the_first_fragment_does_not_make_it_proposed() -> None:
    units = units_for("Project Alpha was proposed, but later approved.")
    ec = validate(
        units,
        modality="proposed",
        predicate="status",
        object_type="text",
        value="proposed",
        currency=None,
        evidence_quotes=["Project Alpha was proposed"],
    )
    assert ec.claim.modality is not Modality.PROPOSED


def test_modality_in_another_sentence_is_not_borrowed() -> None:
    units = units_for("The plan was approved in 2024. Project Alpha has a budget of LKR 6 million.")
    ec = validate(units, value="6000000", evidence_quotes=["budget of LKR 6 million"])
    assert ec.claim.modality is Modality.UNKNOWN and "modality_not_grounded" in ec.issues


def test_modality_conflicting_with_the_sentence_is_rejected() -> None:
    units = units_for("Project Alpha has a proposed budget of LKR 6 million.")
    assert (
        reject_reason(
            units, modality="approved", value="6000000", evidence_quotes=["budget of LKR 6 million"]
        )
        == "modality_conflict"
    )


def test_modality_in_quote_and_same_word_elsewhere_in_sentence() -> None:
    units = units_for("Project Alpha has an approved budget of LKR 6 million, approved in May.")
    ec = validate(units, value="6000000", evidence_quotes=["approved budget of LKR 6 million"])
    assert ec.claim.modality is Modality.APPROVED and "modality_ambiguous" not in ec.issues


def test_modality_from_table_row_cell() -> None:
    c = validate(
        _budget_rows(), value="5500000", evidence_quotes=["Project Alpha", "5500000"]
    ).claim
    assert c.modality is Modality.APPROVED  # the row's "Modality = approved" cell


def test_modality_from_value_column_header_is_cited() -> None:
    cells = xlsx_table(
        DOC, "T", [["Project", "Approved budget (LKR)"], ["Project Alpha", "5000000"]]
    )
    (units,) = plan_sources(cells, DocumentFormat.XLSX).batches
    c = validate(units, value="5000000", evidence_quotes=["Project Alpha", "5000000"]).claim
    assert c.modality is Modality.APPROVED
    assert ("Approved", "B1") in [
        (e.quote, e.location.cell_range) for e in c.evidence if e.role == "header"
    ]


def test_modality_from_another_columns_header_is_not_borrowed() -> None:
    cells = xlsx_table(
        DOC, "T", [["Item", "Proposed cost", "Actual cost"], ["Staff", "3000000", "2800000"]]
    )
    (units,) = plan_sources(cells, DocumentFormat.XLSX).batches
    c = validate(
        units,
        subject="Staff",
        predicate="cost",
        value="2800000",
        currency=None,
        object_type="number",
        modality="actual",
        evidence_quotes=["Staff", "2800000"],
    ).claim
    assert c.modality is Modality.ACTUAL
    assert (
        reject_reason(
            units,
            subject="Staff",
            predicate="cost",
            value="2800000",
            currency=None,
            object_type="number",
            modality="proposed",
            evidence_quotes=["Staff", "2800000"],
        )
        == "modality_conflict"
    )


def test_component_of_an_invalid_date_cannot_ground_a_quantity() -> None:
    units = units_for("Project Alpha was invoiced on 2025-99-99.")
    assert (
        reject_reason(
            units,
            predicate="staff_count",
            object_type="number",
            value="99",
            currency=None,
            modality="unknown",
            evidence_quotes=["Project Alpha was invoiced on 2025-99-99"],
        )
        == "value_not_grounded"
    )


def test_rs_is_ambiguous_without_a_configured_default() -> None:
    """Decision 2026-11-28: 'Rs.' is shared by several rupees; no implicit LKR."""
    units = units_for("Project Alpha received an approved allocation of Rs. 5 million.")
    ec = validate(units, currency="LKR", evidence_quotes=["approved allocation of Rs. 5 million"])
    assert ec.claim.object_type is ObjectType.NUMBER
    assert "currency_not_grounded" in ec.issues
    v = ClaimValidator(units, DOC, "RUN-test", currency_defaults={"Rs": "LKR"})
    c = v.validate_claim(
        LLMClaim.model_validate(
            llm_claim(currency="LKR", evidence_quotes=["approved allocation of Rs. 5 million"])
        )
    ).claim
    assert c.object_type is ObjectType.MONEY and c.object.currency == "LKR"  # type: ignore[union-attr]
