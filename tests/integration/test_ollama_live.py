"""Optional tests against a real local Ollama model.

Skipped unless VERITAS_TEST_OLLAMA=1. Choose the model with VERITAS_OLLAMA_MODEL.
They assert invariants that must hold for *any* model output (everything accepted
is grounded), not specific extractions, so they stay meaningful across models.

    VERITAS_TEST_OLLAMA=1 VERITAS_OLLAMA_MODEL=qwen3:4b pytest -m ollama -s
"""

from __future__ import annotations

import os
from decimal import Decimal

import pytest
from factories import seed_document, span
from sqlalchemy.orm import Session, sessionmaker

from veritas.core.config import Settings
from veritas.core.contracts import DocumentFormat, Modality, Quantity
from veritas.extraction.service import ExtractionService
from veritas.llm.factory import make_provider

pytestmark = [
    pytest.mark.ollama,
    pytest.mark.skipif(
        os.environ.get("VERITAS_TEST_OLLAMA") != "1",
        reason="set VERITAS_TEST_OLLAMA=1 to run against a local Ollama",
    ),
]

TEXT = (
    "Project Alpha has a proposed budget of LKR 7,000,000 for 2026. "
    "Project Alpha has an approved budget of LKR 5,000,000 for 2026. "
    "Project Alpha recorded actual expenditure of LKR 4,500,000 in 2026."
)


def test_live_extraction_invariants(sessions: sessionmaker[Session]) -> None:
    settings = Settings()  # reads VERITAS_OLLAMA_MODEL etc. from the environment
    provider = make_provider(settings)
    doc = "DOC-11111111111a"
    seed_document(sessions, doc, DocumentFormat.PDF, [span(doc, "#/texts/0", TEXT, page=1)])

    run = ExtractionService(provider, sessions, max_retries=2).extract_document(doc)
    print(f"\nmodel={run.model} status={run.status} stats={run.stats()}")
    for ec in run.claims:
        c = ec.claim
        print(
            f"  {c.subject} | {c.predicate} | {c.object} | {c.modality} | "
            f"{c.valid_from}..{c.valid_to} | {ec.issues}"
        )
    for r in run.rejections:
        print(f"  REJECTED {r.reason}: {r.detail}")

    assert run.status != "failed", run.error
    for ec in run.claims:
        for ev in ec.claim.evidence:
            assert ev.quote in TEXT  # grounded by construction
    # Whatever the model said, an accepted value can only carry the modality its
    # own sentence states (or unknown). The three statements never collapse.
    allowed = {
        Decimal("7000000"): {Modality.PROPOSED, Modality.UNKNOWN},
        Decimal("5000000"): {Modality.APPROVED, Modality.UNKNOWN},
        Decimal("4500000"): {Modality.ACTUAL, Modality.UNKNOWN},
    }
    for ec in run.claims:
        obj = ec.claim.object
        if isinstance(obj, Quantity) and obj.value in allowed:
            assert ec.claim.modality in allowed[obj.value], ec.claim
