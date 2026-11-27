"""SQLite persistence (SQLAlchemy 2.0). The relational store is the source of truth.

Milestone 1 tables: ``documents`` and ``document_spans``. Claims, evidence,
entities, conflicts and verifications are added in the milestones that produce
them. Schema is created with ``create_all``; Alembic migrations start once the
schema needs to change under existing data.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
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


def make_engine(url: str) -> Engine:
    parsed = make_url(url)
    if parsed.get_backend_name() == "sqlite" and parsed.database not in (None, "", ":memory:"):
        Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)

    engine = create_engine(url)
    if parsed.get_backend_name() == "sqlite":

        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _record):  # type: ignore[no-untyped-def]
            dbapi_conn.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    return engine


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)
