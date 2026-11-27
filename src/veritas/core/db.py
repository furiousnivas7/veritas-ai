"""SQLite persistence (SQLAlchemy 2.0). The relational store is the source of truth.

Milestone 1 tables: ``documents``, ``document_spans``.
Milestone 2 tables: ``extraction_runs``, ``claims``, ``evidence``, ``entities``,
``entity_aliases``, ``entity_mentions``, ``entity_match_candidates``,
``rejected_extractions``. Conflicts and verifications come in later milestones.

Missing tables are created with ``create_all``. Changes to existing tables are
applied by ``veritas.core.migrations`` (PRAGMA user_version, backup first).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class DocumentRow(Base):
    __tablename__ = "documents"
    __table_args__ = (UniqueConstraint("filename", "version"),)

    document_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    filename: Mapped[str] = mapped_column(String(512), index=True)
    format: Mapped[str] = mapped_column(String(8))
    mime_type: Mapped[str] = mapped_column(String(128))
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    size_bytes: Mapped[int] = mapped_column(Integer)
    version: Mapped[int] = mapped_column(Integer)
    supersedes: Mapped[str | None] = mapped_column(ForeignKey("documents.document_id"))
    status: Mapped[str] = mapped_column(String(16))
    error: Mapped[str | None] = mapped_column(Text)
    storage_path: Mapped[str] = mapped_column(String(1024))
    registered_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    parse_report: Mapped[dict[str, object] | None] = mapped_column(JSON)


class DocumentSpanRow(Base):
    __tablename__ = "document_spans"

    span_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"), index=True
    )
    document_version: Mapped[int] = mapped_column(Integer)
    kind: Mapped[str] = mapped_column(String(16))
    order: Mapped[int] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text)
    parent_span_id: Mapped[str | None] = mapped_column(String(128))
    # Denormalised for querying; the full location is in location_json.
    page_number: Mapped[int | None] = mapped_column(Integer)
    sheet_name: Mapped[str | None] = mapped_column(String(256))
    cell_range: Mapped[str | None] = mapped_column(String(32))
    element_id: Mapped[str | None] = mapped_column(String(64))
    location_json: Mapped[dict[str, object] | None] = mapped_column(JSON)
    missing_provenance: Mapped[str | None] = mapped_column(Text)


# --------------------------------------------------------------- Milestone 2


class ExtractionRunRow(Base):
    """One extraction pass over one document: everything needed to reproduce it."""

    __tablename__ = "extraction_runs"

    run_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.document_id", ondelete="CASCADE"), index=True
    )
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(128))
    options_json: Mapped[dict[str, object]] = mapped_column(JSON)
    prompt_id: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str] = mapped_column(String(16))
    prompt_sha256: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16))
    error: Mapped[str | None] = mapped_column(Text)
    stats_json: Mapped[dict[str, object]] = mapped_column(JSON)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        CheckConstraint("status IN ('completed', 'partial', 'failed')", name="ck_run_status"),
    )


class EntityRow(Base):
    """Names are NOT unique: distinct entities may share a type and normalised name.
    Identity is the opaque ``entity_id``; see ``entities.resolver``."""

    __tablename__ = "entities"
    __table_args__ = (Index("ix_entities_type_name", "entity_type", "normalized_name"),)

    entity_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(512))
    normalized_name: Mapped[str] = mapped_column(String(512), index=True)
    entity_type: Mapped[str] = mapped_column(String(32))


class EntityAliasRow(Base):
    __tablename__ = "entity_aliases"
    __table_args__ = (UniqueConstraint("entity_id", "normalized_alias"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    entity_id: Mapped[str] = mapped_column(
        ForeignKey("entities.entity_id", ondelete="CASCADE"), index=True
    )
    alias: Mapped[str] = mapped_column(String(512))
    normalized_alias: Mapped[str] = mapped_column(String(512), index=True)


class EntityMentionRow(Base):
    __tablename__ = "entity_mentions"
    __table_args__ = (
        UniqueConstraint("run_id", "span_id", "char_start", "char_end", "entity_type"),
    )

    mention_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("extraction_runs.run_id", ondelete="CASCADE"), index=True
    )
    entity_id: Mapped[str | None] = mapped_column(
        ForeignKey("entities.entity_id", ondelete="SET NULL"), index=True
    )
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.document_id"), index=True)
    span_id: Mapped[str] = mapped_column(ForeignKey("document_spans.span_id"), index=True)
    surface: Mapped[str] = mapped_column(String(512))
    normalized: Mapped[str] = mapped_column(String(512), index=True)
    entity_type: Mapped[str] = mapped_column(String(32))
    char_start: Mapped[int] = mapped_column(Integer)
    char_end: Mapped[int] = mapped_column(Integer)
    location_json: Mapped[dict[str, object]] = mapped_column(JSON)


class EntityMatchCandidateRow(Base):
    """Possible identities awaiting review (or the evidence for an automatic merge)."""

    __tablename__ = "entity_match_candidates"
    __table_args__ = (
        UniqueConstraint("mention_id", "entity_id"),
        CheckConstraint("status IN ('merged', 'needs_review', 'rejected')", name="ck_match_status"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    mention_id: Mapped[str] = mapped_column(
        ForeignKey("entity_mentions.mention_id", ondelete="CASCADE"), index=True
    )
    entity_id: Mapped[str] = mapped_column(
        ForeignKey("entities.entity_id", ondelete="CASCADE"), index=True
    )
    status: Mapped[str] = mapped_column(String(16), index=True)
    score: Mapped[float] = mapped_column(Float)
    rule: Mapped[str] = mapped_column(String(32))
    explanation: Mapped[str] = mapped_column(Text)


class ClaimRow(Base):
    __tablename__ = "claims"
    __table_args__ = (
        UniqueConstraint("run_id", "fingerprint"),
        CheckConstraint(
            "modality IN ('proposed', 'approved', 'actual', 'forecast', 'unknown')",
            name="ck_claim_modality",
        ),
        CheckConstraint(
            "extraction_confidence >= 0 AND extraction_confidence <= 1",
            name="ck_claim_confidence",
        ),
    )

    claim_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("extraction_runs.run_id", ondelete="CASCADE"), index=True
    )
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.document_id"), index=True)
    fingerprint: Mapped[str] = mapped_column(String(32))
    subject: Mapped[str] = mapped_column(String(512))
    subject_entity_id: Mapped[str | None] = mapped_column(
        ForeignKey("entities.entity_id", ondelete="SET NULL"), index=True
    )
    predicate: Mapped[str] = mapped_column(String(128), index=True)
    object_type: Mapped[str | None] = mapped_column(String(16))
    object_json: Mapped[object] = mapped_column(JSON)
    # Denormalised for querying numeric claims; the full value is in object_json.
    value_text: Mapped[str | None] = mapped_column(String(64))
    unit: Mapped[str | None] = mapped_column(String(32))
    currency: Mapped[str | None] = mapped_column(String(3))
    modality: Mapped[str] = mapped_column(String(16), index=True)
    valid_from: Mapped[str | None] = mapped_column(String(10))
    valid_to: Mapped[str | None] = mapped_column(String(10))
    time_expression: Mapped[str | None] = mapped_column(String(256))
    negated: Mapped[bool] = mapped_column(Boolean)
    extraction_confidence: Mapped[float] = mapped_column(Float)
    extraction_method: Mapped[str] = mapped_column(String(16))
    issues_json: Mapped[list[str]] = mapped_column(JSON)
    # MetadataDecision list (currency / period provenance); schema v3, existing = []
    metadata_json: Mapped[list[dict[str, object]]] = mapped_column(JSON, server_default="[]")


class EvidenceRow(Base):
    __tablename__ = "evidence"
    __table_args__ = (UniqueConstraint("claim_id", "span_id", "char_start"),)

    evidence_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    claim_id: Mapped[str] = mapped_column(
        ForeignKey("claims.claim_id", ondelete="CASCADE"), index=True
    )
    position: Mapped[int] = mapped_column(Integer)  # order within the claim's evidence
    span_id: Mapped[str] = mapped_column(ForeignKey("document_spans.span_id"), index=True)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.document_id"), index=True)
    document_version: Mapped[int] = mapped_column(Integer)
    quote: Mapped[str] = mapped_column(Text)
    char_start: Mapped[int | None] = mapped_column(Integer)
    char_end: Mapped[int | None] = mapped_column(Integer)
    location_json: Mapped[dict[str, object]] = mapped_column(JSON)
    # quote | scope | header (EvidenceRole); added in schema v2, existing rows = quote
    role: Mapped[str] = mapped_column(String(16), server_default="quote")


class RejectedExtractionRow(Base):
    """Audit trail: every model candidate that failed grounding, with the reason."""

    __tablename__ = "rejected_extractions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(
        ForeignKey("extraction_runs.run_id", ondelete="CASCADE"), index=True
    )
    source_id: Mapped[str] = mapped_column(String(16))
    kind: Mapped[str] = mapped_column(String(16))
    reason: Mapped[str] = mapped_column(String(64), index=True)
    detail: Mapped[str] = mapped_column(Text)
    payload_json: Mapped[dict[str, object]] = mapped_column(JSON)


def make_engine(url: str) -> Engine:
    parsed = make_url(url)
    if parsed.get_backend_name() == "sqlite" and parsed.database not in (None, "", ":memory:"):
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(url)
    if parsed.get_backend_name() == "sqlite":

        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _record):  # type: ignore[no-untyped-def]
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

        # Upgrade existing databases (with a backup) before creating missing tables.
        from veritas.core.migrations import migrate

        db_file = parsed.database
        migrate(engine, Path(db_file) if db_file not in (None, "", ":memory:") else None)

    Base.metadata.create_all(engine)
    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
