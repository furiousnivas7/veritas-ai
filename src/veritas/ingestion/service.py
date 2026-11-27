"""Document ingestion: validate, hash, de-duplicate, version and register."""

from __future__ import annotations

import logging
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session, sessionmaker

from veritas.core.contracts import Document, DocumentFormat, DocumentStatus
from veritas.core.db import DocumentRow
from veritas.ingestion.validation import validate_file

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    document: Document
    duplicate: bool
    storage_path: Path


def new_document_id() -> str:
    return f"DOC-{uuid.uuid4().hex[:12]}"


def row_to_document(row: DocumentRow) -> Document:
    return Document(
        document_id=row.document_id,
        filename=row.filename,
        format=DocumentFormat(row.format),
        mime_type=row.mime_type,
        sha256=row.sha256,
        size_bytes=row.size_bytes,
        version=row.version,
        supersedes=row.supersedes,
        status=DocumentStatus(row.status),
        error=row.error,
        # SQLite does not store tz info; all timestamps are written as UTC.
        registered_at=row.registered_at.replace(tzinfo=row.registered_at.tzinfo or UTC),
    )


class IngestionService:
    """Registers files. Identical content (same SHA-256) is never stored twice;
    new content under an existing filename becomes the next version."""

    def __init__(
        self, session_factory: sessionmaker[Session], storage_dir: Path, max_bytes: int
    ) -> None:
        self._sessions = session_factory
        self._storage = storage_dir
        self._max_bytes = max_bytes

    def ingest(self, path: Path, filename: str | None = None) -> IngestResult:
        """Validate and register ``path``. ``filename`` overrides the stored name
        (e.g. the original name of an uploaded temp file)."""
        vf = validate_file(Path(path), self._max_bytes)
        name = filename or vf.path.name

        with self._sessions.begin() as s:
            existing = s.scalar(select(DocumentRow).where(DocumentRow.sha256 == vf.sha256))
            if existing is not None:
                log.info(
                    "duplicate upload",
                    extra={"document_id": existing.document_id, "sha256": vf.sha256, "file": name},
                )
                return IngestResult(row_to_document(existing), True, Path(existing.storage_path))

            prev = s.scalar(
                select(DocumentRow)
                .where(DocumentRow.filename == name)
                .order_by(DocumentRow.version.desc())
                .limit(1)
            )
            version = (prev.version + 1) if prev else 1

            # Content-addressed copy, so the stored original can never drift from its hash.
            self._storage.mkdir(parents=True, exist_ok=True)
            dest = self._storage / f"{vf.sha256}.{vf.format.value}"
            if not dest.exists():
                shutil.copy2(vf.path, dest)

            doc = Document(
                document_id=new_document_id(),
                filename=name,
                format=vf.format,
                mime_type=vf.mime_type,
                sha256=vf.sha256,
                size_bytes=vf.size_bytes,
                version=version,
                supersedes=prev.document_id if prev else None,
            )
            s.add(
                DocumentRow(
                    **doc.model_dump(exclude={"status"}),
                    status=doc.status.value,
                    storage_path=str(dest),
                )
            )
            log.info(
                "document registered",
                extra={
                    "document_id": doc.document_id,
                    "file": name,
                    "format": doc.format.value,
                    "version": version,
                    "sha256": vf.sha256,
                },
            )
            return IngestResult(doc, False, dest)

    def get(self, document_id: str) -> Document | None:
        with self._sessions() as s:
            row = s.get(DocumentRow, document_id)
            return row_to_document(row) if row else None

    def count(self) -> int:
        with self._sessions() as s:
            return s.scalar(select(func.count()).select_from(DocumentRow)) or 0
