"""Persist and load extraction results. One run is saved in one transaction."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from veritas.core.contracts import (
    Claim,
    Entity,
    EntityType,
    Evidence,
    EvidenceRole,
    MetadataDecision,
    Modality,
    ObjectType,
    Quantity,
    SourceLocation,
)
from veritas.core.db import (
    ClaimRow,
    DocumentSpanRow,
    EntityAliasRow,
    EntityMatchCandidateRow,
    EntityMentionRow,
    EntityRow,
    EvidenceRow,
    ExtractionRunRow,
    RejectedExtractionRow,
)
from veritas.entities.normalize import normalize_name
from veritas.extraction.service import ExtractionRun


class ProvenanceError(ValueError):
    """A claim or mention points to a span that does not exist in the database."""


def _object_json(obj: Any) -> Any:
    if isinstance(obj, Quantity):
        return {"value": format(obj.value, "f"), "unit": obj.unit, "currency": obj.currency}
    return obj


def _object_from_json(data: Any) -> Any:
    if isinstance(data, dict) and "value" in data:
        return Quantity(
            value=Decimal(data["value"]), unit=data.get("unit"), currency=data.get("currency")
        )
    return data


def load_entities(sessions: sessionmaker[Session]) -> list[Entity]:
    with sessions() as s:
        aliases: dict[str, list[str]] = {}
        for a in s.scalars(select(EntityAliasRow)):
            aliases.setdefault(a.entity_id, []).append(a.alias)
        return [
            Entity(
                entity_id=e.entity_id,
                canonical_name=e.canonical_name,
                normalized_name=e.normalized_name,
                entity_type=EntityType(e.entity_type),
                aliases=sorted(aliases.get(e.entity_id, [])),
            )
            for e in s.scalars(select(EntityRow).order_by(EntityRow.entity_id))
        ]


def load_entity_documents(sessions: sessionmaker[Session]) -> dict[str, set[str]]:
    """Entity id -> documents it has mentions in (used for document-scoped merging)."""
    with sessions() as s:
        out: dict[str, set[str]] = {}
        rows = s.execute(
            select(EntityMentionRow.entity_id, EntityMentionRow.document_id)
            .where(EntityMentionRow.entity_id.is_not(None))
            .distinct()
        )
        for entity_id, document_id in rows:
            if entity_id is not None:
                out.setdefault(entity_id, set()).add(document_id)
        return out


def save_run(sessions: sessionmaker[Session], run: ExtractionRun) -> None:
    """Insert the run and everything it produced. Verifies every referenced span
    exists for this document before writing anything (all-or-nothing)."""
    with sessions.begin() as s:
        known = set(
            s.scalars(
                select(DocumentSpanRow.span_id).where(
                    DocumentSpanRow.document_id == run.document_id
                )
            )
        )
        referenced = {ev.span_id for ec in run.claims for ev in ec.claim.evidence}
        referenced |= {m.span_id for m in run.mentions}
        if missing := referenced - known:
            raise ProvenanceError(f"unknown span ids: {sorted(missing)[:5]}")

        s.add(
            ExtractionRunRow(
                run_id=run.run_id,
                document_id=run.document_id,
                provider=run.provider,
                model=run.model,
                options_json=run.options,
                prompt_id=run.prompt_id,
                prompt_version=run.prompt_version,
                prompt_sha256=run.prompt_sha256,
                status=run.status,
                error=run.error,
                stats_json=run.stats(),
                started_at=run.started_at,
                finished_at=run.finished_at,
            )
        )
        s.flush()

        for e in run.entities:
            row = s.get(EntityRow, e.entity_id)
            if row is None:
                s.add(
                    EntityRow(
                        entity_id=e.entity_id,
                        canonical_name=e.canonical_name,
                        normalized_name=e.normalized_name or normalize_name(e.canonical_name),
                        entity_type=(e.entity_type or EntityType.OTHER).value,
                    )
                )
            else:
                row.canonical_name = e.canonical_name
            s.flush()
            have = set(
                s.scalars(
                    select(EntityAliasRow.normalized_alias).where(
                        EntityAliasRow.entity_id == e.entity_id
                    )
                )
            )
            for alias in e.aliases:
                n = normalize_name(alias)
                if n not in have:
                    s.add(EntityAliasRow(entity_id=e.entity_id, alias=alias, normalized_alias=n))
                    have.add(n)
        s.flush()

        for m in run.mentions:
            s.add(
                EntityMentionRow(
                    mention_id=m.mention_id,
                    run_id=run.run_id,
                    entity_id=m.entity_id,
                    document_id=m.document_id,
                    span_id=m.span_id,
                    surface=m.surface,
                    normalized=m.normalized,
                    entity_type=m.entity_type.value,
                    char_start=m.char_start,
                    char_end=m.char_end,
                    location_json=m.location.model_dump(mode="json"),
                )
            )
        s.flush()
        for c in run.candidates:
            s.add(
                EntityMatchCandidateRow(
                    mention_id=c.mention_id,
                    entity_id=c.entity_id,
                    status=c.status.value,
                    score=c.score,
                    rule=c.rule,
                    explanation=c.explanation,
                )
            )

        for ec in run.claims:
            cl = ec.claim
            q = cl.object if isinstance(cl.object, Quantity) else None
            s.add(
                ClaimRow(
                    claim_id=cl.claim_id,
                    run_id=run.run_id,
                    document_id=run.document_id,
                    fingerprint=ec.fingerprint,
                    subject=cl.subject,
                    subject_entity_id=cl.subject_entity_id,
                    predicate=cl.predicate,
                    object_type=cl.object_type.value if cl.object_type else None,
                    object_json=_object_json(cl.object),
                    value_text=format(q.value, "f") if q else None,
                    unit=q.unit if q else None,
                    currency=q.currency if q else None,
                    modality=cl.modality.value,
                    valid_from=cl.valid_from,
                    valid_to=cl.valid_to,
                    time_expression=cl.time_expression,
                    negated=cl.negated,
                    extraction_confidence=cl.extraction_confidence,
                    extraction_method=cl.extraction_method,
                    issues_json=ec.issues,
                    metadata_json=[d.model_dump(mode="json") for d in cl.metadata],
                )
            )
            s.flush()
            for position, ev in enumerate(cl.evidence):
                s.add(
                    EvidenceRow(
                        evidence_id=ev.evidence_id,
                        claim_id=cl.claim_id,
                        position=position,
                        span_id=ev.span_id,
                        document_id=ev.document_id,
                        document_version=ev.document_version,
                        quote=ev.quote,
                        char_start=ev.char_start,
                        char_end=ev.char_end,
                        location_json=ev.location.model_dump(mode="json"),
                        role=ev.role.value,
                    )
                )

        for r in run.rejections:
            s.add(
                RejectedExtractionRow(
                    run_id=run.run_id,
                    source_id=r.source_id,
                    kind=r.kind,
                    reason=r.reason,
                    detail=r.detail,
                    payload_json=r.payload,
                )
            )


def latest_run_id(sessions: sessionmaker[Session], document_id: str) -> str | None:
    with sessions() as s:
        return s.scalar(
            select(ExtractionRunRow.run_id)
            .where(
                ExtractionRunRow.document_id == document_id,
                ExtractionRunRow.status.in_(("completed", "partial")),
            )
            .order_by(ExtractionRunRow.started_at.desc())
            .limit(1)
        )


def load_claims(
    sessions: sessionmaker[Session], document_id: str, run_id: str | None = None
) -> list[Claim]:
    """Claims of ``run_id`` (default: the latest completed or partial run)."""
    run_id = run_id or latest_run_id(sessions, document_id)
    if run_id is None:
        return []
    with sessions() as s:
        rows = s.scalars(
            select(ClaimRow).where(ClaimRow.run_id == run_id).order_by(ClaimRow.claim_id)
        ).all()
        out = []
        for r in rows:
            evs = s.scalars(
                select(EvidenceRow)
                .where(EvidenceRow.claim_id == r.claim_id)
                .order_by(EvidenceRow.position)
            ).all()
            out.append(
                Claim(
                    claim_id=r.claim_id,
                    subject=r.subject,
                    subject_entity_id=r.subject_entity_id,
                    predicate=r.predicate,
                    object=_object_from_json(r.object_json),
                    object_type=ObjectType(r.object_type) if r.object_type else None,
                    modality=Modality(r.modality),
                    valid_from=r.valid_from,
                    valid_to=r.valid_to,
                    time_expression=r.time_expression,
                    negated=r.negated,
                    extraction_confidence=r.extraction_confidence,
                    extraction_method=r.extraction_method,  # type: ignore[arg-type]
                    metadata=[MetadataDecision.model_validate(d) for d in r.metadata_json or []],
                    evidence=[
                        Evidence(
                            evidence_id=e.evidence_id,
                            span_id=e.span_id,
                            document_id=e.document_id,
                            document_version=e.document_version,
                            quote=e.quote,
                            char_start=e.char_start,
                            char_end=e.char_end,
                            location=SourceLocation.model_validate(e.location_json),
                            role=EvidenceRole(e.role),
                        )
                        for e in evs
                    ],
                )
            )
        return out
