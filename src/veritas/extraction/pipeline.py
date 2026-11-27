"""Milestone 2 entry point: extract claims and entities for parsed documents."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from veritas.core.config import Settings
from veritas.core.db import DocumentRow, make_engine, make_session_factory
from veritas.extraction.prompts import get_prompt
from veritas.extraction.repository import load_entities, load_entity_documents, save_run
from veritas.extraction.service import ExtractionRun, ExtractionService
from veritas.llm.base import LLMProvider
from veritas.llm.factory import make_provider


def parsed_document_ids(sessions: sessionmaker[Session]) -> list[str]:
    with sessions() as s:
        return list(
            s.scalars(
                select(DocumentRow.document_id)
                .where(DocumentRow.status == "parsed")
                .order_by(DocumentRow.registered_at)
            )
        )


def run_extraction(
    settings: Settings,
    document_ids: list[str] | None = None,
    *,
    provider: LLMProvider | None = None,
    model: str | None = None,
    sessions: sessionmaker[Session] | None = None,
) -> list[ExtractionRun]:
    """Extract and persist. ``document_ids=None`` means every parsed document; an
    empty list means none. Entities from earlier documents are offered to the
    resolver (merged or queued for review according to the resolution policy).
    Failed runs are persisted too (for audit)."""
    sessions = sessions or make_session_factory(make_engine(settings.resolved_database_url))
    service = ExtractionService(
        provider or make_provider(settings, model),
        sessions,
        max_retries=settings.extraction_max_retries,
        batch_chars=settings.extraction_batch_chars,
        resolution_policy=settings.entity_resolution_policy,
        header_policy=settings.table_header_policy,
        prompt=get_prompt(settings.extraction_prompt_version),
        metadata_enrichment=settings.metadata_enrichment == "on",
        currency_defaults=settings.currency_symbol_defaults,
    )
    runs = []
    targets = parsed_document_ids(sessions) if document_ids is None else document_ids
    for doc_id in targets:
        run = service.extract_document(
            doc_id, load_entities(sessions), load_entity_documents(sessions)
        )
        save_run(sessions, run)
        runs.append(run)
    return runs
