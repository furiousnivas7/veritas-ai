"""Claim and entity extraction for one parsed document.

Flow per document:
  spans -> source batches -> LLM (bounded retries) -> strict schema validation
  -> deterministic grounding -> entity resolution -> one transactional save.

Retry policy (``max_retries`` extra attempts per batch):
* malformed JSON / schema violation -> retry with the error fed back to the model
* timeout / provider error          -> retry the same request
* service unavailable / model missing -> abort the run immediately (not retryable)
A batch that still fails is recorded and skipped; the run becomes ``partial``.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import ValidationError
from sqlalchemy.orm import Session, sessionmaker

from veritas.core.contracts import (
    DocumentFormat,
    Entity,
    EntityMatchCandidate,
    EntityMention,
    utcnow,
)
from veritas.core.db import DocumentRow
from veritas.entities.normalize import find_alias_definitions, normalize_name
from veritas.entities.resolver import EntityResolver, ResolutionPolicy
from veritas.extraction.prompts import DEFAULT_PROMPT, PromptTemplate, retry_message
from veritas.extraction.schema import LLMExtraction, response_schema
from veritas.extraction.sources import HeaderPolicy, SourceUnit, plan_sources
from veritas.extraction.validator import ClaimValidator, ExtractedClaim, Rejected, Rejection
from veritas.llm.base import (
    ChatMessage,
    LLMError,
    LLMProvider,
    LLMRequest,
    LLMUnavailableError,
)
from veritas.parsing.repository import load_spans

log = logging.getLogger(__name__)

RunStatus = Literal["completed", "partial", "failed"]


@dataclass
class BatchOutcome:
    index: int
    attempts: int
    ok: bool
    error: str | None = None
    duration_s: float = 0.0


@dataclass
class ExtractionRun:
    run_id: str
    document_id: str
    provider: str
    model: str
    options: dict[str, Any]
    prompt_id: str
    prompt_version: str
    prompt_sha256: str
    status: RunStatus = "completed"
    error: str | None = None
    started_at: Any = field(default_factory=utcnow)
    finished_at: Any = None
    claims: list[ExtractedClaim] = field(default_factory=list)
    rejections: list[Rejection] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    mentions: list[EntityMention] = field(default_factory=list)
    candidates: list[EntityMatchCandidate] = field(default_factory=list)
    batches: list[BatchOutcome] = field(default_factory=list)
    skipped_missing_provenance: int = 0
    source_units: int = 0
    table_headers: dict[str, Any] = field(default_factory=dict)
    resolution_policy: str = "document_scoped"

    def stats(self) -> dict[str, Any]:
        return {
            "source_units": self.source_units,
            "entity_resolution_policy": self.resolution_policy,
            "batches": len(self.batches),
            "batches_failed": sum(not b.ok for b in self.batches),
            "llm_attempts": sum(b.attempts for b in self.batches),
            "claims_accepted": len(self.claims),
            "claims_rejected": sum(r.kind == "claim" for r in self.rejections),
            "entities_rejected": sum(r.kind == "entity" for r in self.rejections),
            "mentions": len(self.mentions),
            "entities": len(self.entities),
            "match_candidates_for_review": sum(c.status == "needs_review" for c in self.candidates),
            "spans_skipped_missing_provenance": self.skipped_missing_provenance,
            "table_headers": self.table_headers,
            "metadata": _count(
                f"{d.field.value}:{d.origin.value}:{d.status.value}"
                for ec in self.claims
                for d in ec.claim.metadata
            ),
            "rejection_reasons": _count(r.reason for r in self.rejections),
            "llm_seconds": round(sum(b.duration_s for b in self.batches), 2),
            "batch_errors": [
                {"batch": b.index, "attempts": b.attempts, "error": (b.error or "")[:500]}
                for b in self.batches
                if not b.ok
            ],
        }


def _count(items: Any) -> dict[str, int]:
    out: dict[str, int] = {}
    for i in items:
        out[i] = out.get(i, 0) + 1
    return out


class ExtractionService:
    def __init__(
        self,
        provider: LLMProvider,
        sessions: sessionmaker[Session],
        *,
        prompt: PromptTemplate = DEFAULT_PROMPT,
        max_retries: int = 2,
        batch_chars: int = 1800,
        resolution_policy: ResolutionPolicy = "document_scoped",
        header_policy: HeaderPolicy = "detect",
        metadata_enrichment: bool = True,
        currency_defaults: dict[str, str] | None = None,
    ) -> None:
        self.header_policy: HeaderPolicy = header_policy
        self.metadata_enrichment = metadata_enrichment
        self.currency_defaults: dict[str, str] = dict(currency_defaults or {})
        self.provider = provider
        self.resolution_policy: ResolutionPolicy = resolution_policy
        self.sessions = sessions
        self.prompt = prompt
        self.max_retries = max_retries
        self.batch_chars = batch_chars
        self._schema = response_schema()

    def extract_document(
        self,
        document_id: str,
        existing_entities: list[Entity] | None = None,
        entity_documents: dict[str, set[str]] | None = None,
    ) -> ExtractionRun:
        run = ExtractionRun(
            run_id=f"RUN-{uuid.uuid4().hex[:12]}",
            document_id=document_id,
            provider=self.provider.name,
            model=self.provider.model,
            options={
                **self.provider.options(),
                # Extraction configuration that changes results, recorded per run.
                "metadata_enrichment": "on" if self.metadata_enrichment else "off",
                "currency_symbol_defaults": dict(sorted(self.currency_defaults.items())),
                "table_header_policy": self.header_policy,
                "entity_resolution_policy": self.resolution_policy,
            },
            prompt_id=self.prompt.prompt_id,
            prompt_version=self.prompt.version,
            prompt_sha256=self.prompt.sha256,
        )
        with self.sessions() as s:
            doc = s.get(DocumentRow, document_id)
            if doc is None:
                raise KeyError(f"document {document_id} is not registered")
            if doc.status != "parsed":
                raise ValueError(f"document {document_id} is {doc.status}, not parsed")
            fmt = DocumentFormat(doc.format)

        spans = load_spans(self.sessions, document_id)
        plan = plan_sources(spans, fmt, self.batch_chars, self.header_policy)
        run.skipped_missing_provenance = plan.skipped_missing_provenance
        run.source_units = plan.units
        run.table_headers = {
            "policy": self.header_policy,
            "header": plan.tables_with_header,
            "no_header": plan.tables_without_header,
            "ambiguous": plan.tables_header_ambiguous,
            "decisions": plan.header_decisions,
        }

        try:
            self.provider.check()
        except LLMUnavailableError as e:
            return self._fail(run, str(e))

        all_mentions: list[EntityMention] = []
        definitions: list[tuple[str, str]] = []
        model_aliases: list[tuple[str, str]] = []
        seen: set[str] = set()

        for index, batch in enumerate(plan.batches):
            validator = ClaimValidator(
                batch,
                document_id,
                run.run_id,
                enrichment=self.metadata_enrichment,
                currency_defaults=self.currency_defaults,
            )
            try:
                parsed, outcome = self._call(batch, index)
            except LLMUnavailableError as e:
                return self._fail(run, str(e))
            run.batches.append(outcome)
            if parsed is None:
                continue

            for c in parsed.claims:
                try:
                    ec = validator.validate_claim(c, outcome.attempts)
                except Rejected as r:
                    run.rejections.append(
                        Rejection(
                            c.source_id, "claim", r.reason, r.detail, c.model_dump(mode="json")
                        )
                    )
                    continue
                if ec.fingerprint in seen:
                    continue  # the model repeated itself
                seen.add(ec.fingerprint)
                run.claims.append(ec)

            for ent in parsed.entities:
                try:
                    mentions, missing = validator.validate_entity(ent)
                except Rejected as r:
                    run.rejections.append(
                        Rejection(
                            ent.source_id, "entity", r.reason, r.detail, ent.model_dump(mode="json")
                        )
                    )
                    continue
                all_mentions.extend(mentions)
                model_aliases.extend((ent.name, a) for a in ent.aliases if a not in missing)
                for a in missing:
                    run.rejections.append(
                        Rejection(
                            ent.source_id,
                            "entity",
                            "alias_not_in_source",
                            f"alias {a!r} of {ent.name!r} not found",
                            ent.model_dump(mode="json"),
                        )
                    )

            for unit in batch:
                definitions.extend(
                    (d.long_form, d.short_form) for d in find_alias_definitions(unit.text)
                )

        run.resolution_policy = self.resolution_policy
        self._resolve(
            run,
            all_mentions,
            definitions,
            model_aliases,
            EntityResolver(existing_entities or [], entity_documents, self.resolution_policy),
        )
        run.status = "partial" if any(not b.ok for b in run.batches) else "completed"
        if run.batches and all(not b.ok for b in run.batches):
            run.status = "failed"
            run.error = "all batches failed"
        run.finished_at = utcnow()
        log.info("extraction finished", extra={"run_id": run.run_id, **run.stats()})
        return run

    # ------------------------------------------------------------------ internals

    def _call(
        self, batch: list[SourceUnit], index: int
    ) -> tuple[LLMExtraction | None, BatchOutcome]:
        messages: list[ChatMessage] = list(self.prompt.messages(batch))
        started = time.perf_counter()
        last_error = ""
        attempts = 0
        for attempt in range(1, self.max_retries + 2):
            attempts = attempt
            try:
                resp = self.provider.generate(LLMRequest(tuple(messages), self._schema))
            except LLMUnavailableError:
                raise
            except LLMError as e:
                last_error = f"{type(e).__name__}: {e}"
                log.warning(
                    "llm call failed",
                    extra={"batch": index, "attempt": attempt, "error": last_error},
                )
                continue
            try:
                parsed = LLMExtraction.model_validate_json(resp.text)
            except ValidationError as e:
                last_error = _summarize(e)
                log.warning(
                    "invalid model output",
                    extra={"batch": index, "attempt": attempt, "error": last_error},
                )
                messages = [
                    *messages[:2],
                    ChatMessage("assistant", resp.text[:4000]),
                    retry_message(last_error),
                ]
                continue
            return parsed, BatchOutcome(
                index, attempts, True, duration_s=round(time.perf_counter() - started, 3)
            )
        return None, BatchOutcome(
            index, attempts, False, last_error, round(time.perf_counter() - started, 3)
        )

    @staticmethod
    def _resolve(
        run: ExtractionRun,
        mentions: list[EntityMention],
        definitions: list[tuple[str, str]],
        model_aliases: list[tuple[str, str]],
        resolver: EntityResolver,
    ) -> None:
        result = resolver.resolve(mentions, definitions, model_aliases)
        run.entities = list(result.entities.values())
        run.mentions = result.mentions
        run.candidates = result.candidates

        # Link claim subjects to entities only through a mention in the same evidence span.
        by_span: dict[tuple[str, str], set[str]] = {}
        for m in result.mentions:
            if m.entity_id:
                by_span.setdefault((m.span_id, m.normalized), set()).add(m.entity_id)
        for ec in run.claims:
            key_norm = normalize_name(ec.claim.subject)
            ids: set[str] = set()
            for ev in ec.claim.evidence:
                ids |= by_span.get((ev.span_id, key_norm), set())
            if len(ids) == 1:
                ec.claim = ec.claim.model_copy(update={"subject_entity_id": ids.pop()})

    @staticmethod
    def _fail(run: ExtractionRun, error: str) -> ExtractionRun:
        run.status = "failed"
        run.error = error
        run.finished_at = utcnow()
        run.claims, run.mentions, run.entities, run.candidates = [], [], [], []
        log.error("extraction failed", extra={"run_id": run.run_id, "error": error})
        return run


def _summarize(e: ValidationError, limit: int = 8) -> str:
    lines = []
    for err in e.errors()[:limit]:
        loc = ".".join(str(p) for p in err["loc"]) or "(root)"
        lines.append(f"- {loc}: {err['msg']}")
    if e.error_count() > limit:
        lines.append(f"- ... {e.error_count() - limit} more")
    return "\n".join(lines) if lines else json.dumps(str(e))
