from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from pydantic import ValidationError

from veritas.core.contracts import (
    Claim,
    Conflict,
    ConflictType,
    Document,
    DocumentSpan,
    Entity,
    Evidence,
    Modality,
    Quantity,
    Relationship,
    SourceLocation,
    SpanKind,
    Verdict,
    VerificationResult,
)

PDF_LOC: dict[str, Any] = {"format": "pdf", "element_id": "#/texts/4", "page_number": 2}
XLSX_LOC: dict[str, Any] = {
    "format": "xlsx",
    "element_id": "#/tables/0",
    "sheet_name": "Budget",
    "cell_range": "C2",
}
DOCX_LOC: dict[str, Any] = {
    "format": "docx",
    "element_id": "#/texts/2",
    "heading_path": ["Proposal", "Budget request"],
}


def evidence(eid: str = "EV-1", **loc: Any) -> dict[str, Any]:
    return {
        "evidence_id": eid,
        "span_id": "DOC-000000000001:#/texts/4",
        "document_id": "DOC-000000000001",
        "document_version": 1,
        "quote": "approved allocation of LKR 5 million",
        "location": {**PDF_LOC, **loc},
    }


def claim(**overrides: Any) -> dict[str, Any]:
    c: dict[str, Any] = {
        "claim_id": "CLM-1",
        "subject": "Project Alpha",
        "predicate": "budget",
        "object": {"value": "5000000", "unit": "LKR"},
        "modality": "approved",
        "valid_from": "2025-01",
        "extraction_confidence": 0.9,
        "evidence": [evidence()],
    }
    c.update(overrides)
    return c


# ---------------------------------------------------------------- Document


def test_document_valid() -> None:
    d = Document(
        document_id="DOC-0123456789ab",
        filename="a.pdf",
        format="pdf",
        mime_type="application/pdf",
        sha256="a" * 64,
        size_bytes=10,
        version=1,
    )
    assert d.status == "registered"


@pytest.mark.parametrize(
    ("field", "value"), [("sha256", "xyz"), ("document_id", "DOC-1"), ("version", 0)]
)
def test_document_invalid(field: str, value: object) -> None:
    data: dict[str, Any] = {
        "document_id": "DOC-0123456789ab",
        "filename": "a.pdf",
        "format": "pdf",
        "mime_type": "application/pdf",
        "sha256": "a" * 64,
        "size_bytes": 10,
        "version": 1,
        field: value,
    }
    with pytest.raises(ValidationError):
        Document.model_validate(data)


# ---------------------------------------------------------- SourceLocation


@pytest.mark.parametrize("loc", [PDF_LOC, XLSX_LOC, DOCX_LOC])
def test_location_valid(loc: dict[str, Any]) -> None:
    SourceLocation.model_validate(loc)


def test_pdf_requires_page() -> None:
    with pytest.raises(ValidationError, match="page_number"):
        SourceLocation.model_validate({**PDF_LOC, "page_number": None})


def test_xlsx_requires_sheet_and_cell() -> None:
    with pytest.raises(ValidationError, match="sheet_name"):
        SourceLocation.model_validate({**XLSX_LOC, "cell_range": None})


def test_cell_range_normalised() -> None:
    assert SourceLocation.model_validate({**XLSX_LOC, "cell_range": "c2:d5"}).cell_range == "C2:D5"


@pytest.mark.parametrize("bad", ["A0", "1A", "A1:", "AAAA1"])
def test_bad_cell_range(bad: str) -> None:
    with pytest.raises(ValidationError):
        SourceLocation.model_validate({**XLSX_LOC, "cell_range": bad})


def test_format_specific_fields_rejected_elsewhere() -> None:
    with pytest.raises(ValidationError):
        SourceLocation.model_validate({**PDF_LOC, "sheet_name": "S", "cell_range": "A1"})
    with pytest.raises(ValidationError):
        SourceLocation.model_validate({**DOCX_LOC, "page_number": 3})  # DOCX has no pages


def test_table_coords_need_table_ref() -> None:
    with pytest.raises(ValidationError):
        SourceLocation.model_validate({**DOCX_LOC, "table_row": 1, "table_col": 1})
    SourceLocation.model_validate(
        {**DOCX_LOC, "table_ref": "#/tables/0", "table_row": 1, "table_col": 1}
    )


# ------------------------------------------------------------ DocumentSpan


def span(**kw: Any) -> dict[str, Any]:
    s: dict[str, Any] = {
        "span_id": "DOC-000000000001:#/texts/4",
        "document_id": "DOC-000000000001",
        "document_version": 1,
        "kind": "text",
        "text": "Project Alpha received an approved allocation of LKR 5 million.",
        "order": 0,
        "location": PDF_LOC,
    }
    s.update(kw)
    return s


def test_span_needs_location_or_reason() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        DocumentSpan.model_validate(span(location=None))
    with pytest.raises(ValidationError, match="exactly one"):
        DocumentSpan.model_validate(span(missing_provenance="x"))
    DocumentSpan.model_validate(span(location=None, missing_provenance="no prov"))


def test_evidence_from_span_computes_offsets() -> None:
    s = DocumentSpan.model_validate(span())
    ev = Evidence.from_span("EV-1", s, "LKR 5 million")
    assert s.text[ev.char_start : ev.char_end] == "LKR 5 million"  # type: ignore[misc]
    assert ev.location.page_number == 2


def test_evidence_from_unlocated_span_fails() -> None:
    s = DocumentSpan.model_validate(span(location=None, missing_provenance="no prov"))
    with pytest.raises(ValueError, match="no location"):
        Evidence.from_span("EV-1", s)


def test_evidence_quote_must_be_in_span() -> None:
    with pytest.raises(ValueError, match="quote not found"):
        Evidence.from_span("EV-1", DocumentSpan.model_validate(span()), "LKR 7 million")


def test_evidence_offsets_must_match_quote() -> None:
    with pytest.raises(ValidationError, match="quote length"):
        Evidence.model_validate({**evidence(), "char_start": 0, "char_end": 3})


# ------------------------------------------------------------------- Claim


def test_claim_valid() -> None:
    c = Claim.model_validate(claim())
    assert c.modality is Modality.APPROVED
    assert c.object == Quantity(value=Decimal("5000000"), unit="LKR")


def test_claim_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        Claim.model_validate(claim(evidence=[]))


def test_claim_multiple_sources() -> None:
    xlsx_ev = {**evidence("EV-2"), "location": XLSX_LOC}
    c = Claim.model_validate(claim(evidence=[evidence(), xlsx_ev]))
    assert {e.location.format for e in c.evidence} == {"pdf", "xlsx"}


def test_claim_duplicate_evidence_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        Claim.model_validate(claim(evidence=[evidence(), evidence()]))


def test_claim_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Claim.model_validate(claim(made_up=1))


@pytest.mark.parametrize("bad", ["2025-13", "25-01", "2025/01", "January 2025"])
def test_claim_bad_dates(bad: str) -> None:
    with pytest.raises(ValidationError):
        Claim.model_validate(claim(valid_from=bad))


def test_claim_period_order() -> None:
    Claim.model_validate(claim(valid_from="2025-06", valid_to="2025"))
    with pytest.raises(ValidationError, match="earlier"):
        Claim.model_validate(claim(valid_from="2025-06", valid_to="2025-01"))


def test_claim_predicate_snake_case() -> None:
    with pytest.raises(ValidationError):
        Claim.model_validate(claim(predicate="Approved Budget"))


# ------------------------------------------------- Entity / Relationship


def test_entity_and_relationship() -> None:
    Entity(
        entity_id="ENT-USJ-001",
        canonical_name="University of Sri Jayewardenepura",
        aliases=["USJ", "Sri Jayewardenepura University"],
    )
    Relationship(relationship_id="R1", source_id="ENT-1", relation="same_as", target_id="ENT-2")
    with pytest.raises(ValidationError):
        Relationship(relationship_id="R1", source_id="E", relation="same_as", target_id="E")


# ----------------------------------------------------------------- Conflict


def conflict(**kw: Any) -> dict[str, Any]:
    c: dict[str, Any] = {
        "conflict_id": "CF-1",
        "claim_a_id": "CLM-1",
        "claim_b_id": "CLM-2",
        "verdict": "contradicted",
        "conflict_type": "numerical",
        "explanation": "Approved budget is 5M in the report and 5.5M in the workbook.",
        "detection_confidence": 0.8,
        "method": "rule:numeric",
    }
    c.update(kw)
    return c


def test_conflict_valid() -> None:
    assert Conflict.model_validate(conflict()).conflict_type is ConflictType.NUMERICAL


def test_contradiction_needs_type() -> None:
    with pytest.raises(ValidationError, match="conflict_type"):
        Conflict.model_validate(conflict(conflict_type=None))


def test_contextual_difference_has_no_type() -> None:
    Conflict.model_validate(conflict(verdict="contextually_different", conflict_type=None))
    with pytest.raises(ValidationError):
        Conflict.model_validate(conflict(verdict="contextually_different"))


def test_conflict_rejects_single_claim_verdict() -> None:
    with pytest.raises(ValidationError):
        Conflict.model_validate(conflict(verdict="supported", conflict_type=None))


def test_conflict_needs_explanation_and_two_claims() -> None:
    with pytest.raises(ValidationError):
        Conflict.model_validate(conflict(explanation=""))
    with pytest.raises(ValidationError):
        Conflict.model_validate(conflict(claim_b_id="CLM-1"))


# ------------------------------------------------------- VerificationResult


def vr(**kw: Any) -> dict[str, Any]:
    v: dict[str, Any] = {
        "verification_id": "V-1",
        "claim_id": "CLM-1",
        "verdict": "supported",
        "verification_confidence": 0.7,
        "method": "nli",
        "supporting_evidence_ids": ["EV-1"],
    }
    v.update(kw)
    return v


def test_supported_needs_evidence() -> None:
    VerificationResult.model_validate(vr())
    with pytest.raises(ValidationError):
        VerificationResult.model_validate(vr(supporting_evidence_ids=[]))


def test_refuted_needs_refuting_evidence() -> None:
    with pytest.raises(ValidationError):
        VerificationResult.model_validate(vr(verdict="refuted"))


def test_explicit_abstention() -> None:
    v = VerificationResult.model_validate(
        vr(
            verdict="insufficient_evidence",
            supporting_evidence_ids=[],
            abstention_reason="no span mentions Project Alpha's 2026 budget",
        )
    )
    assert v.abstained
    with pytest.raises(ValidationError, match="abstention_reason"):
        VerificationResult.model_validate(vr(verdict="requires_human_review"))
    with pytest.raises(ValidationError, match="abstention_reason"):
        VerificationResult.model_validate(vr(abstention_reason="not an abstention"))


def test_verification_rejects_pair_verdict() -> None:
    with pytest.raises(ValidationError):
        VerificationResult.model_validate(vr(verdict=Verdict.CONTRADICTED))


def test_confidences_are_separate_fields() -> None:
    # Extraction confidence lives on Claim; verification confidence on VerificationResult.
    assert "extraction_confidence" in Claim.model_fields
    assert "verification_confidence" in VerificationResult.model_fields
    assert "verification_confidence" not in Claim.model_fields


def test_span_kind_values() -> None:
    assert SpanKind.TABLE_CELL == "table_cell"
