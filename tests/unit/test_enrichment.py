"""Deterministic currency / period enrichment (approved 2026-11-28).

Every accepted value must cite located evidence inside the claim's own scope; model
output is never silently overwritten; ambiguity yields no value plus a reason; and
enrichment "off" reproduces the previous behaviour."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from factories import llm_claim, llm_json, seed_document, span, xlsx_table
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from veritas.core.config import Settings
from veritas.core.contracts import (
    Claim,
    DocumentFormat,
    MetadataDecision,
    ObjectType,
    Quantity,
)
from veritas.core.db import ClaimRow, ExtractionRunRow
from veritas.extraction import enrichment as enrich
from veritas.extraction.pipeline import run_extraction
from veritas.extraction.prompts import CLAIM_EXTRACTION_V1, DEFAULT_PROMPT, PROMPTS, get_prompt
from veritas.extraction.repository import load_claims, save_run
from veritas.extraction.schema import LLMClaim
from veritas.extraction.service import ExtractionService
from veritas.extraction.sources import SourceUnit, plan_sources
from veritas.extraction.validator import ClaimValidator, ExtractedClaim
from veritas.llm.mock_provider import MockProvider

DOC = "DOC-000000000001"


def prose(*texts: str, heading: list[str] | None = None) -> list[SourceUnit]:
    spans = [span(DOC, f"#/texts/{i}", t, order=i, page=1, heading_path=heading)
             for i, t in enumerate(texts)]  # fmt: skip
    return plan_sources(spans, DocumentFormat.PDF).batches[0]


def table(
    rows: list[list[str]], sheet: str = "Budget", ref: str = "#/tables/0"
) -> list[SourceUnit]:
    return plan_sources(xlsx_table(DOC, sheet, rows, ref=ref), DocumentFormat.XLSX).batches[0]


def run(units: list[SourceUnit], *, enrichment: bool = True, defaults: dict[str, str] | None = None,
        **claim: Any) -> ExtractedClaim:  # fmt: skip
    v = ClaimValidator(units, DOC, "RUN-t", enrichment=enrichment, currency_defaults=defaults)
    return v.validate_claim(LLMClaim.model_validate(llm_claim(**claim)))


def money(
    units: list[SourceUnit], quotes: list[str], value: str = "7000000", **kw: Any
) -> ExtractedClaim:
    kw.setdefault("currency", None)
    kw.setdefault("modality", "unknown")
    return run(units, value=value, evidence_quotes=quotes, **kw)


def decision(c: Claim, field: str) -> MetadataDecision:
    (d,) = [d for d in c.metadata if d.field == field]
    return d


def cited(c: Claim, d: MetadataDecision) -> list[str]:
    by_id = {e.evidence_id: e for e in c.evidence}
    return [by_id[i].quote for i in d.evidence_ids]


def assert_traceable(c: Claim) -> None:
    """Every decision's evidence exists in the claim and is an exact source slice."""
    ids = {e.evidence_id for e in c.evidence}
    for d in c.metadata:
        assert set(d.evidence_ids) <= ids
        if d.status == "accepted":
            assert d.evidence_ids
    for e in c.evidence:
        assert e.char_start is not None and e.char_end is not None
        assert e.char_end - e.char_start == len(e.quote)


# ======================================================== prose currency


def test_model_silent_currency_is_filled_from_the_quote() -> None:
    """The real qwen3:4b case: object_type money, currency null, 'LKR 7 million' quoted."""
    q = "Project Alpha had a proposed budget of LKR 7 million for 2025."
    ec = money(prose(q), [q], modality="proposed")
    c = ec.claim
    assert c.object == Quantity(value=Decimal("7000000"), currency="LKR")
    assert c.object_type is ObjectType.MONEY
    d = decision(c, "currency")
    assert (d.origin, d.status, d.reason, d.model_value) == ("enrichment", "accepted",
                                                             "attached_to_value", None)  # fmt: skip
    assert cited(c, d) == [q]  # already covered by the quote; no extra slice
    assert "currency_not_grounded" not in ec.issues and "currency_absent" not in ec.issues
    assert_traceable(c)


def test_currency_just_outside_the_quote_is_cited_as_scope() -> None:
    text = "Project Alpha received LKR 7 million in 2025."
    c = money(prose(text), ["Project Alpha received", "7 million"]).claim
    assert c.object.currency == "LKR"  # type: ignore[union-attr]
    d = decision(c, "currency")
    (lkr,) = [e for e in c.evidence if e.evidence_id in d.evidence_ids]
    assert (lkr.quote, lkr.role, lkr.char_start) == ("LKR", "scope", text.index("LKR"))


@pytest.mark.parametrize(("phrase", "code"), [
    ("LKR 7 million", "LKR"), ("7 million LKR", "LKR"), ("7 million Sri Lankan rupees", "LKR"),
    ("US$ 7 million", "USD"), ("€7 million", "EUR"), ("£7 million", "GBP"), ("₹7 million", "INR"),
])  # fmt: skip
def test_recognised_currency_forms(phrase: str, code: str) -> None:
    q = f"Project Alpha received {phrase}."
    assert money(prose(q), [q]).claim.object.currency == code  # type: ignore[union-attr]


@pytest.mark.parametrize("phrase", ["Rs. 7 million", "$7 million", "7 million rupees",
                                    "7 million dollars", "¥7 million"])  # fmt: skip
def test_ambiguous_symbol_resolves_to_nothing(phrase: str) -> None:
    q = f"Project Alpha received {phrase}."
    ec = money(prose(q), [q])
    assert ec.claim.object_type is ObjectType.NUMBER
    d = decision(ec.claim, "currency")
    assert (d.status, d.reason) == ("ambiguous", "ambiguous_currency_symbol")
    assert d.candidate_value and "currency_ambiguous" in ec.issues


def test_explicit_default_resolves_ambiguous_symbol() -> None:
    q = "Project Alpha received Rs. 7 million."
    c = money(prose(q), [q], defaults={"Rs.": "LKR"}).claim
    assert c.object.currency == "LKR"  # type: ignore[union-attr]
    assert decision(c, "currency").origin == "enrichment"


# ========================================================= table currency


def test_currency_in_the_value_cell_itself() -> None:
    units = table([["Project", "Amount"], ["Project Alpha", "LKR 5,500,000"]])
    c = money(units, ["Project Alpha", "LKR 5,500,000"], value="5500000").claim
    assert c.object.currency == "LKR"  # type: ignore[union-attr]
    assert decision(c, "currency").reason == "attached_to_value"


def test_currency_in_the_value_columns_header_is_cited_as_header() -> None:
    units = table([["Project", "Amount (LKR)"], ["Project Alpha", "5500000"]])
    c = money(units, ["Project Alpha", "5500000"], value="5500000").claim
    d = decision(c, "currency")
    assert (c.object.currency, d.origin, d.reason) == ("LKR", "enrichment", "value_column_header")  # type: ignore[union-attr]
    (h,) = [e for e in c.evidence if e.evidence_id in d.evidence_ids]
    assert (h.quote, h.role, h.location.cell_range) == ("LKR", "header", "B1")


# ============================================================ not borrowed


@pytest.mark.parametrize(
    ("units_fn", "quotes", "value"),
    [
        # another sentence
        (lambda: prose("Budgets are reported in LKR. Project Alpha received 7 million."),
         ["Project Alpha received 7 million"], "7000000"),
        # another amount in the same quote
        (lambda: prose("Project Alpha received LKR 5 million and 300 laptops."),
         ["Project Alpha received LKR 5 million and 300 laptops"], "300"),
        # a heading
        (lambda: prose("Project Alpha received 7 million.", heading=["Budget (LKR)"]),
         ["Project Alpha received 7 million"], "7000000"),
        # another column's header
        (lambda: table([["Project", "Staff", "Notes (LKR)"], ["Project Alpha", "42", "pending"]]),
         ["Project Alpha", "42", "pending"], "42"),
        # the sheet name
        (lambda: table([["Project", "Amount"], ["Project Alpha", "5"]], sheet="LKR budget"),
         ["Project Alpha", "5"], "5"),
    ],
)  # fmt: skip
def test_currency_is_never_borrowed(units_fn: Any, quotes: list[str], value: str) -> None:
    ec = money(units_fn(), quotes, value=value)
    assert ec.claim.object_type is ObjectType.NUMBER
    assert decision(ec.claim, "currency").status == "absent"
    assert "currency_absent" in ec.issues


def test_currency_from_another_row_is_not_borrowed() -> None:
    cells = [c for c in xlsx_table(DOC, "S", [["Project", "Amount"], ["Project Alpha", "5"],
                                              ["Project Beta", "LKR 9"]])
             if c.location and c.location.table_row]  # fmt: skip
    hdr = {c.location.table_col: c for c in xlsx_table(DOC, "S", [["Project", "Amount"]])}  # type: ignore[union-attr]
    unit = SourceUnit("S1", "table_row", "rows", "", tuple(cells),
                      tuple(hdr[c.location.table_col].text for c in cells),  # type: ignore[union-attr]
                      tuple(hdr[c.location.table_col] for c in cells))  # type: ignore[union-attr]  # fmt: skip
    ec = money([unit], ["Project Alpha", "5"], value="5")
    assert decision(ec.claim, "currency").status == "absent"


# ===================================================== multiple currencies


def test_each_amount_gets_its_own_currency() -> None:
    q = "Project Alpha received USD 5 million and LKR 1.5 billion."
    units = prose(q)
    assert money(units, [q], value="5000000").claim.object.currency == "USD"  # type: ignore[union-attr]
    assert money(units, [q], value="1500000000").claim.object.currency == "LKR"  # type: ignore[union-attr]


def test_two_currencies_on_one_value_are_ambiguous() -> None:
    units = table([["Project", "Amount (USD)"], ["Project Alpha", "LKR 5,500,000"]])
    ec = money(units, ["Project Alpha", "LKR 5,500,000"], value="5500000")
    d = decision(ec.claim, "currency")
    assert (d.status, d.candidate_value) == ("ambiguous", "LKR,USD")
    assert ec.claim.object_type is ObjectType.NUMBER


def test_document_with_lkr_and_usd_tables(sessions: sessionmaker[Session]) -> None:
    doc = "DOC-0000000000c1"
    spans = xlsx_table(doc, "Local", [["Project", "Amount (LKR)"], ["Project Alpha", "5500000"]])
    spans += xlsx_table(doc, "Grants", [["Project", "Amount (USD)"], ["Project Alpha", "20000"]],
                        ref="#/tables/1")  # fmt: skip
    seed_document(sessions, doc, DocumentFormat.XLSX, spans)
    response = llm_json([
        llm_claim(source_id="S1", value="5500000", currency=None, modality="unknown",
                  evidence_quotes=["Project Alpha", "5500000"]),
        llm_claim(source_id="S2", value="20000", currency=None, modality="unknown",
                  evidence_quotes=["Project Alpha", "20000"]),
    ])  # fmt: skip
    r = ExtractionService(MockProvider([response]), sessions).extract_document(doc)
    assert sorted(ec.claim.object.currency for ec in r.claims) == ["LKR", "USD"]  # type: ignore[union-attr]


# =================================================================== periods


@pytest.mark.parametrize(("sentence", "expected"), [
    ("Project Alpha received LKR 7 million in 2025.", ("2025", "2025")),
    ("Project Alpha received LKR 7 million in March 2025.", ("2025-03", "2025-03")),
    ("Project Alpha received LKR 7 million on 15 January 2025.", ("2025-01-15", "2025-01-15")),
    ("Project Alpha received LKR 7 million from January to March 2025.", ("2025-01", "2025-03")),
    ("Project Alpha received LKR 7 million between 2024 and 2026.", ("2024", "2026")),
    ("Project Alpha received LKR 7 million for FY 2024/25.", ("2024", "2025")),
])  # fmt: skip
def test_period_is_read_at_its_stated_precision(sentence: str, expected: tuple[str, str]) -> None:
    c = money(prose(sentence), [sentence]).claim
    assert (c.valid_from, c.valid_to) == expected
    d = decision(c, "period")
    assert (d.origin, d.status, d.value) == (
        "enrichment",
        "accepted",
        f"{expected[0]}..{expected[1]}",
    )
    assert c.time_expression and c.time_expression in sentence
    assert_traceable(c)


def test_year_stays_a_year() -> None:
    q = "Project Alpha received LKR 7 million for 2025."
    c = money(prose(q), [q]).claim
    assert (c.valid_from, c.valid_to) == ("2025", "2025")  # never "2025-01-01"


def test_period_just_outside_the_quote_is_cited_as_scope() -> None:
    text = "Project Alpha received LKR 7 million in March 2025."
    c = money(prose(text), ["Project Alpha received LKR 7 million"]).claim
    d = decision(c, "period")
    (e,) = [e for e in c.evidence if e.evidence_id in d.evidence_ids]
    assert (e.quote, e.role) == ("in March 2025", "scope")


def test_several_dates_in_the_sentence_are_ambiguous() -> None:
    q = "Project Alpha was approved in January 2025 and received LKR 7 million in June 2025."
    ec = money(prose(q), ["received LKR 7 million"], value="7000000", subject="Project Alpha")
    assert (ec.claim.valid_from, ec.claim.valid_to) == (None, None)
    d = decision(ec.claim, "period")
    assert (d.status, d.reason) == ("ambiguous", "several_periods_in_evidence")
    assert "period_ambiguous" in ec.issues


def test_quoted_date_is_preferred_over_another_date_in_the_sentence() -> None:
    q = "Project Alpha was approved in January 2025 and received LKR 7 million in June 2025."
    c = money(prose(q), ["Project Alpha was approved", "received LKR 7 million in June 2025"]).claim
    assert (c.valid_from, c.valid_to) == ("2025-06", "2025-06")


def test_date_in_another_sentence_is_never_used() -> None:
    units = prose("The report covers 2024. Project Alpha received LKR 7 million.")
    ec = money(units, ["Project Alpha received LKR 7 million"])
    assert ec.claim.valid_from is None and decision(ec.claim, "period").status == "absent"


def test_date_in_heading_is_never_used() -> None:
    units = prose("Project Alpha received LKR 7 million.", heading=["Annual Report 2025"])
    assert money(units, ["Project Alpha received LKR 7 million"]).claim.valid_from is None


def test_date_valued_claim_is_not_given_its_own_date_as_period() -> None:
    q = "Project Alpha was completed in June 2025."
    c = run(prose(q), predicate="completion_date", object_type="date", value="2025-06",
            currency=None, modality="unknown", evidence_quotes=[q]).claim  # fmt: skip
    assert (c.object, c.valid_from) == ("2025-06", None)


def test_year_that_is_the_value_is_not_a_period() -> None:
    q = "Project Alpha was launched in 2024."
    c = run(prose(q), predicate="launch_year", object_type="number", value="2024",
            currency=None, modality="unknown", evidence_quotes=[q]).claim  # fmt: skip
    assert c.valid_from is None


# ===================================================== ambiguous / impossible


def test_ambiguous_numeric_date_is_reduced_to_its_year() -> None:
    q = "Project Alpha received LKR 7 million on 01/02/2025."
    ec = money(prose(q), [q])
    assert (ec.claim.valid_from, ec.claim.valid_to) == ("2025", "2025")
    d = decision(ec.claim, "period")
    assert (d.status, d.reason) == ("accepted", "precision_reduced_to_year")
    assert "period_precision_reduced" in ec.issues


def test_unambiguous_numeric_date_keeps_day_precision() -> None:
    q = "Project Alpha received LKR 7 million on 15/01/2025."
    assert money(prose(q), [q]).claim.valid_from == "2025-01-15"


@pytest.mark.parametrize(("q", "reason"), [
    ("Project Alpha received LKR 7 million on 2025-02-30.", "invalid_date_in_evidence"),
    ("Project Alpha received LKR 7 million on 31/02/2025.", "invalid_date_in_evidence"),
    ("Project Alpha received LKR 7 million in Q1 2025.", "unsupported_time_expression"),
    ("Project Alpha received LKR 7 million since 2024.", "unsupported_time_expression"),
])  # fmt: skip
def test_impossible_or_unsupported_dates_give_no_period(q: str, reason: str) -> None:
    ec = money(prose(q), [q])
    assert ec.claim.valid_from is None
    assert decision(ec.claim, "period").reason == reason


# ==================================================================== tables


def test_row_date_column_is_the_period_with_header_cited() -> None:
    units = table([["Project", "Amount (LKR)", "Date"], ["Project Alpha", "5500000", "2025-01"]])
    c = money(units, ["Project Alpha", "5500000"], value="5500000").claim
    assert (c.valid_from, c.valid_to) == ("2025-01", "2025-01")
    assert sorted(cited(c, decision(c, "period"))) == ["2025-01", "Date"]


def test_start_and_end_columns_give_a_range() -> None:
    units = table([["Project", "Amount (LKR)", "Start", "End"],
                   ["Project Alpha", "5500000", "2025-01", "2025-06"]])  # fmt: skip
    c = money(units, ["Project Alpha", "5500000"], value="5500000").claim
    assert (c.valid_from, c.valid_to) == ("2025-01", "2025-06")


def test_unlabelled_date_in_an_ambiguous_table_is_not_a_period() -> None:
    units = table([["Project Alpha", "pending", "2025-01"], ["Project Beta", "5", "2025-03"]])
    beta = next(u for u in units if "Project Beta" in u.text)
    ec = money(units, ["Project Beta", "5"], value="5", subject="Project Beta",
               source_id=beta.source_id)  # fmt: skip
    assert ec.claim.valid_from is None
    assert decision(ec.claim, "period").reason == "row_date_unlabelled"


def test_year_in_value_column_header_is_not_a_period() -> None:
    """Approved decision D3: 'Budget 2025' as a header never supplies a period (v1)."""
    units = table([["Project", "Budget 2025"], ["Project Alpha", "5500000"]])
    assert money(units, ["Project Alpha", "5500000"], value="5500000").claim.valid_from is None


def test_year_in_sheet_name_is_not_a_period() -> None:
    units = table([["Project", "Amount"], ["Project Alpha", "5"]], sheet="Budget 2025")
    assert money(units, ["Project Alpha", "5"], value="5").claim.valid_from is None


# ======================================================== model conflicts


def test_model_currency_contradicting_the_evidence_is_not_overwritten() -> None:
    q = "Project Alpha received USD 7 million."
    ec = money(prose(q), [q], currency="LKR")
    assert ec.claim.object_type is ObjectType.NUMBER  # neither LKR nor USD is used
    d = decision(ec.claim, "currency")
    assert (d.status, d.model_value, d.candidate_value) == ("conflict", "LKR", "USD")
    assert "currency_conflict" in ec.issues


def test_model_currency_agreeing_with_the_evidence_keeps_model_origin() -> None:
    q = "Project Alpha received LKR 7 million."
    d = decision(money(prose(q), [q], currency="LKR").claim, "currency")
    assert (d.origin, d.status, d.model_value) == ("model", "accepted", "LKR")


def test_model_period_contradicting_the_evidence_is_not_substituted() -> None:
    q = "Project Alpha received LKR 7 million in 2025."
    ec = money(prose(q), [q], valid_from="2026", valid_to="2026", time_expression="in 2026")
    assert (ec.claim.valid_from, ec.claim.valid_to) == (None, None)
    d = decision(ec.claim, "period")
    assert (d.status, d.model_value, d.candidate_value) == ("conflict", "2026..2026", "2025..2025")
    assert "time_not_grounded" in ec.issues


def test_model_period_supported_by_the_evidence_keeps_model_origin() -> None:
    q = "Project Alpha received LKR 7 million in 2025."
    c = money(prose(q), [q], valid_from="2025", valid_to="2025", time_expression="in 2025").claim
    d = decision(c, "period")
    assert (d.origin, d.value, c.time_expression) == ("model", "2025..2025", "in 2025")


# ================================================================== missing


def test_nothing_in_evidence_gives_absent_decisions() -> None:
    q = "Project Alpha received 7 million."
    ec = money(prose(q), [q])
    c = ec.claim
    assert (c.object_type, c.valid_from) == (ObjectType.NUMBER, None)
    assert decision(c, "currency").status == "absent" and decision(c, "period").status == "absent"
    assert_traceable(c)


# ============================================================== off mode


@pytest.mark.parametrize("q", [
    "Project Alpha had a proposed budget of LKR 7 million for 2025.",
    "Project Alpha received 7 million.",
    "Project Alpha received USD 7 million in 2025.",
])  # fmt: skip
def test_off_reproduces_previous_behaviour(q: str) -> None:
    """Off: no enrichment, no decisions, previous issue names and confidence."""
    units = prose(q)
    off = money(units, [q], enrichment=False)
    assert off.claim.metadata == []
    assert off.claim.valid_from is None
    assert off.claim.object_type is ObjectType.NUMBER  # model gave no currency
    assert off.issues == ["currency_not_grounded"]
    assert off.claim.extraction_confidence == round(0.9 * 0.7, 3)


def test_on_and_off_agree_when_model_supplies_supported_metadata() -> None:
    q = "Project Alpha received LKR 7 million in 2025."
    kw = dict(currency="LKR", valid_from="2025", valid_to="2025", time_expression="in 2025")
    on, off = money(prose(q), [q], **kw), money(prose(q), [q], enrichment=False, **kw)
    assert on.issues == off.issues and on.fingerprint == off.fingerprint
    strip = {"metadata": True, "claim_id": True, "evidence": {"__all__": {"evidence_id"}}}
    assert on.claim.model_dump(exclude=strip) == off.claim.model_dump(exclude=strip)


# ============================================== persistence and run options


def test_decisions_survive_save_and_reload(sessions: sessionmaker[Session]) -> None:
    doc = "DOC-0000000000c2"
    text = "Project Alpha had a proposed budget of LKR 7 million for 2025."
    seed_document(sessions, doc, DocumentFormat.PDF, [span(doc, "#/texts/0", text)])
    response = llm_json([llm_claim(value="7000000", currency=None, modality="proposed",
                                   evidence_quotes=[text])])  # fmt: skip
    r = ExtractionService(MockProvider([response]), sessions).extract_document(doc)
    save_run(sessions, r)
    (c,) = load_claims(sessions, doc)
    assert c == r.claims[0].claim
    assert {(d.field, d.origin, d.status) for d in c.metadata} == {
        ("currency", "enrichment", "accepted"),
        ("period", "enrichment", "accepted"),
    }
    with sessions() as s:
        row = s.scalars(select(ClaimRow).where(ClaimRow.claim_id == c.claim_id)).one()
        assert (row.currency, row.valid_from) == ("LKR", "2025")
        assert {d["field"] for d in row.metadata_json} == {"currency", "period"}
        run_row = s.get(ExtractionRunRow, r.run_id)
        assert run_row is not None
        assert run_row.options_json["metadata_enrichment"] == "on"
        assert run_row.options_json["currency_symbol_defaults"] == {}
        assert run_row.stats_json["metadata"] == {
            "currency:enrichment:accepted": 1,
            "period:enrichment:accepted": 1,
        }


def test_settings_drive_enrichment_defaults_and_prompt(
    sessions: sessionmaker[Session], settings: Settings
) -> None:
    doc = "DOC-0000000000c3"
    text = "Project Alpha received Rs. 7 million in 2025."
    seed_document(sessions, doc, DocumentFormat.PDF, [span(doc, "#/texts/0", text)])
    reply = llm_json([llm_claim(value="7000000", currency=None, modality="unknown",
                                evidence_quotes=[text])])  # fmt: skip
    cfg = settings.model_copy(update={"currency_symbol_defaults": {"rs": "LKR"}})
    (r,) = run_extraction(cfg, [doc], provider=MockProvider([reply]), sessions=sessions)
    assert r.claims[0].claim.object.currency == "LKR"  # type: ignore[union-attr]
    assert r.options["currency_symbol_defaults"] == {"rs": "LKR"}
    assert r.prompt_version == "1.1.0"
    off = settings.model_copy(update={"metadata_enrichment": "off",
                                      "extraction_prompt_version": "1.0.0"})  # fmt: skip
    (r2,) = run_extraction(off, [doc], provider=MockProvider([reply]), sessions=sessions)
    assert r2.claims[0].claim.metadata == [] and r2.prompt_version == "1.0.0"
    assert r2.options["metadata_enrichment"] == "off"


@pytest.mark.parametrize(
    ("raw", "norm"),
    [
        ({"Rs.": "lkr"}, {"rs": "LKR"}),
        ({"$": "USD", "Rupee": "INR"}, {"$": "USD", "rupees": "INR"}),
    ],
)
def test_currency_default_settings_are_normalised(
    raw: dict[str, str], norm: dict[str, str]
) -> None:
    assert Settings(currency_symbol_defaults=raw).currency_symbol_defaults == norm


@pytest.mark.parametrize("bad", [{"LKR": "LKR"}, {"Rs": "XYZ"}, {"euro": "EUR"}])
def test_invalid_currency_defaults_are_refused(bad: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        Settings(currency_symbol_defaults=bad)


def test_currency_defaults_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("VERITAS_CURRENCY_SYMBOL_DEFAULTS", '{"Rs": "LKR"}')
    monkeypatch.setenv("VERITAS_METADATA_ENRICHMENT", "off")
    s = Settings()
    assert (s.currency_symbol_defaults, s.metadata_enrichment) == ({"rs": "LKR"}, "off")


# =================================================================== prompt


def test_prompt_versions() -> None:
    assert DEFAULT_PROMPT.version == "1.1.0"
    assert set(PROMPTS) == {
        ("claim_entity_extraction", "1.0.0"),
        ("claim_entity_extraction", "1.1.0"),
    }
    assert "or its context" not in DEFAULT_PROMPT.system
    assert "or its context" in CLAIM_EXTRACTION_V1.system  # preserved verbatim
    with pytest.raises(ValueError):
        get_prompt("9.9.9")


def test_prompt_1_0_0_is_byte_identical_to_the_recorded_runs() -> None:
    """Pinned to the sha256 recorded by the 2026-11-28 qwen3:4b smoke run."""
    assert CLAIM_EXTRACTION_V1.sha256 == (
        "0db825885c05c1459aeef0a684bc53e2e6e6f65d3f0e918a8d5f60846a5925cc"
    )


# ===================================================== enrichment parsers


@pytest.mark.parametrize(("label", "role"), [("Date", "period"), ("Year", "period"),
                                             ("Start date", "start"), ("End", "end"),
                                             ("Amount", None), ("Budget 2025", None)])  # fmt: skip
def test_header_time_role(label: str, role: str | None) -> None:
    assert enrich.header_time_role(label) == role


def test_bare_year_without_cue_is_not_a_period() -> None:
    assert enrich.find_periods("2025 participants attended; 42 staff joined.").periods == []
