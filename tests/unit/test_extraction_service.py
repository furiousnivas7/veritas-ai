"""Extraction service end to end with a mocked model and a real SQLite database."""

from __future__ import annotations

import json

import pytest
from factories import llm_claim, llm_entity, llm_json, seed_document, span, xlsx_table
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from veritas.core.config import Settings
from veritas.core.contracts import DocumentFormat, MatchStatus, Modality
from veritas.core.db import (
    ClaimRow,
    DocumentSpanRow,
    EntityMatchCandidateRow,
    EntityMentionRow,
    EntityRow,
    EvidenceRow,
    ExtractionRunRow,
    RejectedExtractionRow,
)
from veritas.extraction.pipeline import run_extraction
from veritas.extraction.prompts import DEFAULT_PROMPT
from veritas.extraction.repository import ProvenanceError, load_claims, load_entities, save_run
from veritas.extraction.service import ExtractionService
from veritas.llm.base import LLMResponseError, LLMTimeoutError, LLMUnavailableError
from veritas.llm.mock_provider import MockProvider
from veritas.parsing.repository import save_parse
from veritas.parsing.service import ParsedDocument, ParseReport

PDF = "DOC-00000000000a"
TEXT = "Project Alpha received an approved allocation of LKR 5 million in January 2025."
GOOD = llm_json(
    [llm_claim(valid_from="2025-01", valid_to="2025-01", time_expression="in January 2025")],
    [llm_entity()],
)


@pytest.fixture
def pdf_doc(sessions: sessionmaker[Session]) -> str:
    seed_document(
        sessions,
        PDF,
        DocumentFormat.PDF,
        [span(PDF, "#/texts/0", TEXT, page=2, heading_path=["2. Project Alpha"])],
    )
    return PDF


def service(
    sessions: sessionmaker[Session], provider: MockProvider, retries: int = 2
) -> ExtractionService:
    return ExtractionService(provider, sessions, max_retries=retries)


def count(sessions: sessionmaker[Session], table: type) -> int:
    with sessions() as s:
        return s.scalar(select(func.count()).select_from(table)) or 0


# ------------------------------------------------------------------ happy path


def test_extract_and_persist(sessions: sessionmaker[Session], pdf_doc: str) -> None:
    run = service(sessions, MockProvider([GOOD])).extract_document(pdf_doc)
    assert run.status == "completed"
    save_run(sessions, run)

    (claim,) = load_claims(sessions, pdf_doc)
    assert claim == run.claims[0].claim  # round trip through SQLite is lossless
    assert claim.evidence[0].location.page_number == 2
    assert claim.subject_entity_id is not None  # linked through the mention in the same span

    with sessions() as s:
        row = s.get(ExtractionRunRow, run.run_id)
        assert row is not None
        assert (row.model, row.prompt_id, row.prompt_version) == (
            "mock-model",
            DEFAULT_PROMPT.prompt_id,
            DEFAULT_PROMPT.version,
        )
        assert row.prompt_sha256 == DEFAULT_PROMPT.sha256
        assert row.stats_json["claims_accepted"] == 1


def test_prompt_contains_source_ids_and_schema(
    sessions: sessionmaker[Session], pdf_doc: str
) -> None:
    mock = MockProvider([GOOD])
    service(sessions, mock).extract_document(pdf_doc)
    (req,) = mock.requests
    assert "[S1]" in req.messages[1].content and TEXT in req.messages[1].content
    assert "context: 2. Project Alpha" in req.messages[1].content
    assert req.json_schema is not None and "claims" in json.dumps(req.json_schema)


def test_multiple_claims_from_one_element(sessions: sessionmaker[Session]) -> None:
    doc = "DOC-00000000000b"
    text = (
        "Project Alpha has a proposed budget of LKR 7,000,000 for 2026. "
        "Project Alpha has an approved budget of LKR 5,000,000 for 2026. "
        "Project Alpha recorded actual expenditure of LKR 4,500,000 in 2026."
    )
    seed_document(sessions, doc, DocumentFormat.PDF, [span(doc, "#/texts/0", text)])
    response = llm_json(
        [
            llm_claim(
                modality="proposed",
                value="7000000",
                valid_from="2026",
                valid_to="2026",
                time_expression="for 2026",
                evidence_quotes=["proposed budget of LKR 7,000,000"],
            ),
            llm_claim(
                modality="approved",
                value="5000000",
                valid_from="2026",
                valid_to="2026",
                time_expression="for 2026",
                evidence_quotes=["approved budget of LKR 5,000,000"],
            ),
            llm_claim(
                predicate="expenditure",
                modality="actual",
                value="4500000",
                valid_from="2026",
                valid_to="2026",
                time_expression="in 2026",
                evidence_quotes=["recorded actual expenditure of LKR 4,500,000"],
            ),
        ]
    )
    run = service(sessions, MockProvider([response])).extract_document(doc)
    save_run(sessions, run)
    got = {(c.predicate, c.modality) for c in load_claims(sessions, doc)}
    assert got == {
        ("budget", Modality.PROPOSED),
        ("budget", Modality.APPROVED),
        ("expenditure", Modality.ACTUAL),
    }


def test_repeated_claim_is_deduplicated(sessions: sessionmaker[Session], pdf_doc: str) -> None:
    run = service(sessions, MockProvider([llm_json([llm_claim(), llm_claim()])])).extract_document(
        pdf_doc
    )
    assert len(run.claims) == 1


# --------------------------------------------------------- malformed responses


@pytest.mark.parametrize(
    "bad",
    [
        "not json at all",
        '{"claims": [',  # truncated
        llm_json([{k: v for k, v in llm_claim().items() if k != "modality"}]),  # missing field
        llm_json([llm_claim(modality="approvedd")]),  # invalid modality
        llm_json([llm_claim(extra_field=1)]),  # unknown field
        llm_json([llm_claim(confidence=1.7)]),  # out of range
        json.dumps({"claims": []}),  # missing entities key
    ],
)
def test_malformed_output_is_retried_with_feedback(
    sessions: sessionmaker[Session], pdf_doc: str, bad: str
) -> None:
    mock = MockProvider([bad, GOOD])
    run = service(sessions, mock).extract_document(pdf_doc)
    assert run.status == "completed"
    assert run.batches[0].attempts == 2
    assert "retried" in run.claims[0].issues
    retry = mock.requests[1].messages
    assert retry[-2].role == "assistant" and retry[-1].role == "user"
    assert "rejected because" in retry[-1].content


def test_retries_are_bounded(sessions: sessionmaker[Session], pdf_doc: str) -> None:
    mock = MockProvider(["{}", "{}", "{}", GOOD])
    run = service(sessions, mock, retries=2).extract_document(pdf_doc)
    assert len(mock.requests) == 3  # 1 + 2 retries, never the 4th
    assert run.status == "failed" and run.claims == []
    assert run.batches[0].error and "claims" in run.batches[0].error
    save_run(sessions, run)  # failed runs are persisted for audit
    with sessions() as s:
        assert s.get(ExtractionRunRow, run.run_id).status == "failed"  # type: ignore[union-attr]


# ------------------------------------------------------- timeouts / outages


def test_timeout_then_success(sessions: sessionmaker[Session], pdf_doc: str) -> None:
    run = service(sessions, MockProvider([LLMTimeoutError("slow"), GOOD])).extract_document(pdf_doc)
    assert run.status == "completed" and run.batches[0].attempts == 2


def test_timeouts_exhausted(sessions: sessionmaker[Session], pdf_doc: str) -> None:
    mock = MockProvider([LLMTimeoutError("slow")] * 3)
    run = service(sessions, mock).extract_document(pdf_doc)
    assert run.status == "failed"
    assert "LLMTimeoutError" in (run.batches[0].error or "")


def test_provider_error_is_retried(sessions: sessionmaker[Session], pdf_doc: str) -> None:
    run = service(sessions, MockProvider([LLMResponseError("500"), GOOD])).extract_document(pdf_doc)
    assert run.status == "completed"


def test_unavailable_service_fails_fast(sessions: sessionmaker[Session], pdf_doc: str) -> None:
    mock = MockProvider([GOOD], available=False)
    run = service(sessions, mock).extract_document(pdf_doc)
    assert run.status == "failed" and "unavailable" in (run.error or "")
    assert mock.requests == []
    save_run(sessions, run)
    assert count(sessions, ClaimRow) == 0


def test_outage_mid_run_is_not_retried(sessions: sessionmaker[Session], pdf_doc: str) -> None:
    mock = MockProvider([LLMUnavailableError("connection refused"), GOOD])
    run = service(sessions, mock).extract_document(pdf_doc)
    assert run.status == "failed" and len(mock.requests) == 1


def test_partial_run_keeps_successful_batches(sessions: sessionmaker[Session]) -> None:
    doc = "DOC-00000000000c"
    long_a = TEXT + " " + "x" * 300
    long_b = "Project Beta was completed in June 2025. " + "y" * 300
    seed_document(
        sessions,
        doc,
        DocumentFormat.PDF,
        [span(doc, "#/texts/0", long_a, order=0), span(doc, "#/texts/1", long_b, order=1)],
    )
    svc = ExtractionService(
        MockProvider([GOOD, "bad", "bad", "bad"]), sessions, max_retries=2, batch_chars=400
    )
    run = svc.extract_document(doc)
    assert len(run.batches) == 2
    assert run.status == "partial" and len(run.claims) == 1


# --------------------------------------------------------- rejections / audit


def test_rejections_are_recorded(sessions: sessionmaker[Session], pdf_doc: str) -> None:
    response = llm_json(
        [
            llm_claim(),
            llm_claim(value="9000000"),  # not in source
            llm_claim(source_id="S7"),  # invalid reference
            llm_claim(evidence_quotes=["on page 12"]),  # invented quote
        ],
        [llm_entity(name="Project Omega")],
    )
    run = service(sessions, MockProvider([response])).extract_document(pdf_doc)
    save_run(sessions, run)
    assert len(run.claims) == 1
    with sessions() as s:
        reasons = sorted(s.scalars(select(RejectedExtractionRow.reason)))
    assert reasons == [
        "entity_not_in_source",
        "quote_not_in_source",
        "unknown_source",
        "value_not_grounded",
    ]


def test_save_rejects_claims_citing_unknown_spans(
    sessions: sessionmaker[Session], pdf_doc: str
) -> None:
    run = service(sessions, MockProvider([GOOD])).extract_document(pdf_doc)
    ec = run.claims[0]
    bad_ev = ec.claim.evidence[0].model_copy(update={"span_id": f"{PDF}:#/texts/999"})
    ec.claim = ec.claim.model_copy(update={"evidence": [bad_ev]})
    with pytest.raises(ProvenanceError):
        save_run(sessions, run)
    assert count(sessions, ExtractionRunRow) == 0  # nothing partial was written


def test_spans_without_provenance_are_never_sent(sessions: sessionmaker[Session]) -> None:
    doc = "DOC-00000000000d"
    seed_document(
        sessions,
        doc,
        DocumentFormat.PDF,
        [
            span(doc, "#/texts/0", TEXT, order=0),
            span(doc, "#/texts/1", "Project Beta received LKR 2 million.", order=1, located=False),
        ],
    )
    mock = MockProvider([GOOD])
    run = service(sessions, mock).extract_document(doc)
    assert run.skipped_missing_provenance == 1
    assert "Project Beta" not in mock.requests[0].messages[1].content


def test_unparsed_document_is_refused(sessions: sessionmaker[Session]) -> None:
    with pytest.raises(KeyError):
        service(sessions, MockProvider([GOOD])).extract_document("DOC-ffffffffffff")


# ------------------------------------------------------------ constraints


def test_claim_fingerprint_unique_per_run(sessions: sessionmaker[Session], pdf_doc: str) -> None:
    run = service(sessions, MockProvider([GOOD])).extract_document(pdf_doc)
    dup = run.claims[0]
    run.claims.append(
        type(dup)(
            dup.claim.model_copy(
                update={
                    "claim_id": "CLM-dup",
                    "evidence": [
                        dup.claim.evidence[0].model_copy(update={"evidence_id": "EV-dup"})
                    ],
                }
            ),
            dup.source_id,
            [],
            dup.fingerprint,
        )
    )
    with pytest.raises(IntegrityError):
        save_run(sessions, run)


def test_reparse_after_extraction_is_blocked(sessions: sessionmaker[Session], pdf_doc: str) -> None:
    """Spans cited as evidence cannot be silently replaced (foreign keys)."""
    save_run(sessions, service(sessions, MockProvider([GOOD])).extract_document(pdf_doc))
    evidence_before = count(sessions, EvidenceRow)
    assert evidence_before >= 1
    report = ParseReport(
        document_id=pdf_doc,
        filename="x.pdf",
        format="pdf",
        sha256="0" * 64,
        status="parsed",
        parser="test",
    )
    with pytest.raises(IntegrityError):
        save_parse(sessions, ParsedDocument(report=report, spans=[]))
    assert count(sessions, EvidenceRow) == evidence_before


# --------------------------------------------------------------- entities


USJ_D1, USJ_D2 = "DOC-0000000000e1", "DOC-0000000000e2"


def _seed_usj_documents(sessions: sessionmaker[Session]) -> MockProvider:
    seed_document(
        sessions,
        USJ_D1,
        DocumentFormat.PDF,
        [
            span(
                USJ_D1,
                "#/texts/0",
                "The Research Office of the University of Sri Jayewardenepura (USJ) reports.",
            )
        ],
    )
    seed_document(sessions, USJ_D2, DocumentFormat.DOCX, [span(USJ_D2, "#/texts/0",
        "USJ will host the project. Sri Jayewardenepura University signed it.",
        fmt=DocumentFormat.DOCX, page=None)])  # fmt: skip

    def responder(req):  # type: ignore[no-untyped-def]
        if "Research Office" in req.messages[1].content:
            return llm_json(
                [],
                [
                    llm_entity(
                        name="University of Sri Jayewardenepura",
                        entity_type="organization",
                        aliases=["USJ"],
                    )
                ],
            )
        return llm_json([], [llm_entity(name="USJ", entity_type="organization"),
                             llm_entity(name="Sri Jayewardenepura University",
                                        entity_type="organization")])  # fmt: skip

    return MockProvider(responder=responder)


def _review(sessions: sessionmaker[Session]) -> list[EntityMatchCandidateRow]:
    with sessions() as s:
        return list(s.scalars(select(EntityMatchCandidateRow).where(
            EntityMatchCandidateRow.status == MatchStatus.NEEDS_REVIEW.value)))  # fmt: skip


def test_entities_across_documents_are_queued_for_review_by_default(
    sessions: sessionmaker[Session], settings: Settings
) -> None:
    """Default policy: the alias is proven in doc 1 only, so doc 2's 'USJ' is a
    review candidate, not a silent merge."""
    run_extraction(settings, [USJ_D1, USJ_D2], provider=_seed_usj_documents(sessions),
                   sessions=sessions)  # fmt: skip
    ents = load_entities(sessions)
    usj = next(e for e in ents if e.canonical_name == "University of Sri Jayewardenepura")
    assert "USJ" in usj.aliases  # defined and merged within doc 1
    doc2_usj = [e for e in ents if e.canonical_name == "USJ"]
    assert len(doc2_usj) == 1 and doc2_usj[0].entity_id != usj.entity_id
    rules = {(c.rule, c.entity_id) for c in _review(sessions)}
    assert ("exact_cross_document", usj.entity_id) in rules
    assert ("token_set", usj.entity_id) in rules  # 'Sri Jayewardenepura University'
    assert "Sri Jayewardenepura University" in {e.canonical_name for e in ents}


def test_entities_across_documents_merge_under_global_name_match(
    sessions: sessionmaker[Session], settings: Settings
) -> None:
    baseline = settings.model_copy(update={"entity_resolution_policy": "global_name_match"})
    run_extraction(baseline, [USJ_D1, USJ_D2], provider=_seed_usj_documents(sessions),
                   sessions=sessions)  # fmt: skip
    ents = load_entities(sessions)
    assert "USJ" not in {e.canonical_name for e in ents}  # merged by name
    assert "Sri Jayewardenepura University" in {e.canonical_name for e in ents}  # still not
    with sessions() as s:
        runs = s.scalars(select(ExtractionRunRow)).all()
    assert {r.stats_json["entity_resolution_policy"] for r in runs} == {"global_name_match"}


def test_entities_with_identical_names_can_coexist(sessions: sessionmaker[Session]) -> None:
    with sessions.begin() as s:
        s.add_all([
            EntityRow(entity_id="ENT-a", canonical_name="Project Alpha",
                      normalized_name="project alpha", entity_type="project"),
            EntityRow(entity_id="ENT-b", canonical_name="Project Alpha",
                      normalized_name="project alpha", entity_type="project"),
        ])  # fmt: skip
    assert count(sessions, EntityRow) == 2


def test_same_named_projects_in_two_documents_stay_separate(
    sessions: sessionmaker[Session], settings: Settings
) -> None:
    d1, d2 = "DOC-0000000000a1", "DOC-0000000000a2"
    for d in (d1, d2):
        seed_document(sessions, d, DocumentFormat.PDF, [span(d, "#/texts/0", TEXT)])
    run_extraction(settings, [d1, d2], provider=MockProvider(responder=lambda r: GOOD),
                   sessions=sessions)  # fmt: skip
    alphas = [e for e in load_entities(sessions) if e.normalized_name == "project alpha"]
    assert len(alphas) == 2
    claims = load_claims(sessions, d1) + load_claims(sessions, d2)
    assert len({c.subject_entity_id for c in claims}) == 2


# ------------------------------------------------------ mention uniqueness


def _mention_row(run_id: str, mention_id: str, start: int, end: int) -> EntityMentionRow:
    loc = {"format": "pdf", "element_id": "#/texts/0", "page_number": 2}
    return EntityMentionRow(mention_id=mention_id, run_id=run_id, entity_id=None,
                            document_id=PDF, span_id=f"{PDF}:#/texts/0", surface=TEXT[start:end],
                            normalized=TEXT[start:end].lower(), entity_type="project",
                            char_start=start, char_end=end, location_json=loc)  # fmt: skip


def test_mentions_with_same_start_but_different_end_are_distinct(
    sessions: sessionmaker[Session], pdf_doc: str
) -> None:
    run = service(sessions, MockProvider([llm_json()])).extract_document(pdf_doc)
    save_run(sessions, run)
    with sessions.begin() as s:
        s.add(_mention_row(run.run_id, "MEN-short", 0, 7))  # "Project"
        s.add(_mention_row(run.run_id, "MEN-long", 0, 13))  # "Project Alpha"
    assert count(sessions, EntityMentionRow) == 2


def test_duplicate_mention_record_is_rejected(
    sessions: sessionmaker[Session], pdf_doc: str
) -> None:
    run = service(sessions, MockProvider([llm_json()])).extract_document(pdf_doc)
    save_run(sessions, run)
    with pytest.raises(IntegrityError):
        with sessions.begin() as s:
            s.add(_mention_row(run.run_id, "MEN-1", 0, 13))
            s.add(_mention_row(run.run_id, "MEN-2", 0, 13))


def test_model_repeating_an_entity_does_not_break_saving(
    sessions: sessionmaker[Session], pdf_doc: str
) -> None:
    response = llm_json([llm_claim()], [llm_entity(), llm_entity()])
    run = service(sessions, MockProvider([response])).extract_document(pdf_doc)
    save_run(sessions, run)
    assert count(sessions, EntityMentionRow) == 1


# ------------------------------------------------------------- pipeline


def test_explicit_empty_document_list_processes_nothing(
    sessions: sessionmaker[Session], settings: Settings, pdf_doc: str
) -> None:
    mock = MockProvider([GOOD])
    assert run_extraction(settings, [], provider=mock, sessions=sessions) == []
    assert mock.requests == [] and count(sessions, ExtractionRunRow) == 0


def test_none_processes_all_parsed_documents(
    sessions: sessionmaker[Session], settings: Settings, pdf_doc: str
) -> None:
    runs = run_extraction(settings, None, provider=MockProvider([GOOD]), sessions=sessions)
    assert [r.document_id for r in runs] == [pdf_doc]


def test_xlsx_document_claims(sessions: sessionmaker[Session]) -> None:
    doc = "DOC-0000000000f1"
    seed_document(
        sessions,
        doc,
        DocumentFormat.XLSX,
        xlsx_table(
            doc,
            "Budget",
            [
                ["Project", "Modality", "Amount (LKR)", "Date"],
                ["Project Alpha", "approved", "5500000", "2025-01"],
                ["Project Alpha", "actual", "4800000", "2025-09"],
            ],
        ),
    )
    response = llm_json(
        [
            llm_claim(
                value="5500000",
                valid_from="2025-01",
                valid_to="2025-01",
                time_expression="2025-01",
                evidence_quotes=["5500000", "approved"],
            ),
            llm_claim(
                source_id="S2",
                predicate="expenditure",
                modality="actual",
                value="4800000",
                valid_from="2025-09",
                valid_to="2025-09",
                time_expression="2025-09",
                evidence_quotes=["4800000", "actual"],
            ),
        ]
    )
    run = service(sessions, MockProvider([response])).extract_document(doc)
    save_run(sessions, run)
    got = {
        (c.predicate, c.modality.value, c.evidence[0].location.cell_range)
        for c in load_claims(sessions, doc)
    }
    assert got == {("budget", "approved", "C2"), ("expenditure", "actual", "C3")}


def test_omitted_optional_fields_default_to_null(
    sessions: sessionmaker[Session], pdf_doc: str
) -> None:
    """JSON-mode models omit null fields; that is not a missing required field."""
    optional = {"unit", "currency", "valid_from", "valid_to", "time_expression", "negated"}
    sparse = {
        k: v for k, v in llm_claim(currency="LKR").items() if k not in optional - {"currency"}
    }
    # Enrichment off: the omitted fields simply default (schema behaviour).
    off = ExtractionService(MockProvider([llm_json([sparse])]), sessions, metadata_enrichment=False)
    run = off.extract_document(pdf_doc)
    assert run.status == "completed" and run.batches[0].attempts == 1
    c = run.claims[0].claim
    assert (c.valid_from, c.negated, c.metadata) == (None, False, [])
    # Enrichment on (default): the period stated in the quoted sentence is filled,
    # and the decision records that it came from the evidence, not the model.
    run = service(sessions, MockProvider([llm_json([sparse])])).extract_document(pdf_doc)
    c = run.claims[0].claim
    assert (c.valid_from, c.valid_to) == ("2025-01", "2025-01")
    (period,) = [d for d in c.metadata if d.field == "period"]
    assert (period.origin, period.status, period.model_value) == ("enrichment", "accepted", None)


def test_claim_json_output_is_readable(sessions: sessionmaker[Session], pdf_doc: str) -> None:
    run = service(sessions, MockProvider([GOOD])).extract_document(pdf_doc)
    assert run.claims[0].claim.model_dump(mode="json")["object"]["value"] == "5000000"


def test_two_claims_citing_the_same_text_are_both_saved(
    sessions: sessionmaker[Session], pdf_doc: str
) -> None:
    """Evidence ids are per claim: a shared quote must not collide on save."""
    quote = ["Project Alpha received an approved allocation of LKR 5 million in January 2025"]
    response = llm_json(
        [
            llm_claim(evidence_quotes=quote),
            llm_claim(
                predicate="allocation_date",
                object_type="date",
                value="2025-01",
                currency=None,
                evidence_quotes=quote,
            ),
        ]
    )
    run = service(sessions, MockProvider([response])).extract_document(pdf_doc)
    assert len(run.claims) == 2
    save_run(sessions, run)
    claims = load_claims(sessions, pdf_doc)
    assert len(claims) == 2
    assert {e.span_id for c in claims for e in c.evidence} == {f"{PDF}:#/texts/0"}


def test_header_evidence_role_round_trips(sessions: sessionmaker[Session]) -> None:
    doc = "DOC-0000000000f2"
    seed_document(
        sessions,
        doc,
        DocumentFormat.XLSX,
        xlsx_table(doc, "Budget", [["Project", "Amount (LKR)"], ["Project Alpha", "5500000"]]),
    )
    response = llm_json(
        [
            llm_claim(
                value="5500000", modality="unknown", evidence_quotes=["Project Alpha", "5500000"]
            )
        ]
    )
    run = service(sessions, MockProvider([response])).extract_document(doc)
    save_run(sessions, run)
    (c,) = load_claims(sessions, doc)
    assert c == run.claims[0].claim
    assert [(e.role, e.location.cell_range, e.quote) for e in c.evidence] == [
        ("quote", "A2", "Project Alpha"),
        ("quote", "B2", "5500000"),
        ("header", "B1", "LKR"),
    ]


def test_run_stats_report_table_header_decisions(sessions: sessionmaker[Session]) -> None:
    doc = "DOC-0000000000f3"
    spans = xlsx_table(doc, "A", [["Project", "Amount (LKR)"], ["Project Alpha", "5"]])
    spans += xlsx_table(
        doc, "B", [["Project Alpha", "pending"], ["Project Beta", "5"]], ref="#/tables/1"
    )
    seed_document(sessions, doc, DocumentFormat.XLSX, spans)
    run = service(sessions, MockProvider([llm_json()])).extract_document(doc)
    th = run.stats()["table_headers"]
    assert {k: th[k] for k in ("policy", "header", "no_header", "ambiguous")} == {
        "policy": "detect",
        "header": 1,
        "no_header": 0,
        "ambiguous": 1,
    }
    decisions = {d["table_ref"]: d for d in th["decisions"]}
    assert decisions["#/tables/0"]["status"] == "header"
    assert decisions["#/tables/0"]["header_span_ids"] == [
        f"{doc}:#/tables/0:r0c0",
        f"{doc}:#/tables/0:r0c1",
    ]
    assert decisions["#/tables/1"]["status"] == "ambiguous"
    assert decisions["#/tables/1"]["header_span_ids"] == []
    assert "not column labels" in decisions["#/tables/1"]["reason"]
    assert run.source_units == 3  # 1 data row + both rows of the ambiguous table


def test_header_policy_never_from_settings(
    sessions: sessionmaker[Session], settings: Settings
) -> None:
    doc = "DOC-0000000000f4"
    seed_document(
        sessions,
        doc,
        DocumentFormat.XLSX,
        xlsx_table(doc, "A", [["Project", "Amount"], ["Alpha", "5"]]),
    )
    mock = MockProvider(responder=lambda r: llm_json())
    (run,) = run_extraction(
        settings.model_copy(update={"table_header_policy": "never"}),
        [doc],
        provider=mock,
        sessions=sessions,
    )
    assert run.stats()["table_headers"]["policy"] == "never"
    assert run.source_units == 2
    assert "row: Project | Amount" in mock.requests[0].messages[1].content


def test_table_header_audit_is_persisted_in_run_stats(sessions: sessionmaker[Session]) -> None:
    """The header decisions must survive save_run: read back from SQLite, not run.stats()."""
    doc = "DOC-0000000000f5"
    header_table = xlsx_table(doc, "Budget", [["Project", "Amount (LKR)"], ["Project Alpha", "5"]])
    ambiguous_table = xlsx_table(
        doc, "Notes", [["Project Alpha", "pending"], ["Project Beta", "5"]], ref="#/tables/1"
    )
    seed_document(sessions, doc, DocumentFormat.XLSX, header_table + ambiguous_table)

    run = service(sessions, MockProvider([llm_json()])).extract_document(doc)
    save_run(sessions, run)

    with sessions() as s:  # fresh session: the row as stored in the database
        row = s.get(ExtractionRunRow, run.run_id)
        assert row is not None
        stored = row.stats_json["table_headers"]
        raw = s.execute(
            select(ExtractionRunRow.stats_json).where(ExtractionRunRow.run_id == run.run_id)
        ).scalar_one()
    assert raw["table_headers"] == stored

    assert isinstance(stored, dict)
    assert stored["policy"] == "detect"
    assert (stored["header"], stored["no_header"], stored["ambiguous"]) == (1, 0, 1)
    decisions = {d["table_ref"]: d for d in stored["decisions"]}
    assert set(decisions) == {"#/tables/0", "#/tables/1"}
    assert decisions["#/tables/0"] == {
        "table_ref": "#/tables/0",
        "status": "header",
        "reason": "first-row cells are all column labels over a value column",
        "header_span_ids": [f"{doc}:#/tables/0:r0c0", f"{doc}:#/tables/0:r0c1"],
    }
    assert decisions["#/tables/1"] == {
        "table_ref": "#/tables/1",
        "status": "ambiguous",
        "reason": "first-row cells are not column labels: ['Project Alpha', 'pending']",
        "header_span_ids": [],  # nothing withheld: both rows stay data
    }

    # The withheld header cells are real, located spans of this document.
    with sessions() as s:
        ids = set(decisions["#/tables/0"]["header_span_ids"])
        found = s.scalars(select(DocumentSpanRow).where(DocumentSpanRow.span_id.in_(ids))).all()
    assert {r.span_id for r in found} == ids
    assert all(r.location_json is not None for r in found)
