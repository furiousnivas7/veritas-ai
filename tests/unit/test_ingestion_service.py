from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session, sessionmaker

from veritas.core.db import DocumentRow
from veritas.ingestion.service import IngestionService
from veritas.ingestion.validation import UnsupportedFormatError


def test_registers_metadata(
    ingestion: IngestionService, sessions: sessionmaker[Session], sample_files: dict[str, Path]
) -> None:
    src = sample_files["pdf"]
    res = ingestion.ingest(src)
    doc = res.document

    assert not res.duplicate
    assert re.fullmatch(r"DOC-[0-9a-f]{12}", doc.document_id)
    assert doc.filename == src.name
    assert doc.format == "pdf"
    assert doc.mime_type == "application/pdf"
    assert doc.sha256 == hashlib.sha256(src.read_bytes()).hexdigest()
    assert doc.size_bytes == src.stat().st_size
    assert doc.version == 1 and doc.supersedes is None
    assert doc.status == "registered"

    with sessions() as s:
        row = s.get(DocumentRow, doc.document_id)
        assert row is not None
        assert row.sha256 == doc.sha256
        assert row.status == "registered"
    assert ingestion.get(doc.document_id) == doc


def test_stored_copy_is_content_addressed(
    ingestion: IngestionService, sample_files: dict[str, Path]
) -> None:
    res = ingestion.ingest(sample_files["xlsx"])
    assert res.storage_path.name == f"{res.document.sha256}.xlsx"
    assert res.storage_path.read_bytes() == sample_files["xlsx"].read_bytes()


def test_duplicate_upload_detected(
    ingestion: IngestionService, sample_files: dict[str, Path], tmp_path: Path
) -> None:
    first = ingestion.ingest(sample_files["docx"])
    again = ingestion.ingest(sample_files["docx"])
    # Same bytes under a different name is still a duplicate.
    renamed = tmp_path / "copy_of_proposal.docx"
    renamed.write_bytes(sample_files["docx"].read_bytes())
    third = ingestion.ingest(renamed)

    assert again.duplicate and third.duplicate
    assert again.document.document_id == first.document.document_id
    assert third.document.document_id == first.document.document_id
    assert ingestion.count() == 1


def test_changed_content_creates_new_version(
    ingestion: IngestionService,
    sessions: sessionmaker[Session],
    tmp_path: Path,
    sample_files: dict[str, Path],
) -> None:
    import openpyxl

    p = tmp_path / "budget.xlsx"
    p.write_bytes(sample_files["xlsx"].read_bytes())
    v1 = ingestion.ingest(p).document

    wb = openpyxl.load_workbook(p)
    wb["Budget"]["C2"] = 6_000_000
    wb.save(p)
    v2 = ingestion.ingest(p).document

    assert (v1.version, v2.version) == (1, 2)
    assert v2.supersedes == v1.document_id
    assert v2.document_id != v1.document_id and v2.sha256 != v1.sha256
    with sessions() as s:
        versions = s.scalars(
            select(DocumentRow.version).where(DocumentRow.filename == "budget.xlsx")
        ).all()
    assert sorted(versions) == [1, 2]


def test_filename_override(ingestion: IngestionService, sample_files: dict[str, Path]) -> None:
    doc = ingestion.ingest(sample_files["pdf"], filename="Annual Report.pdf").document
    assert doc.filename == "Annual Report.pdf"


def test_rejected_file_is_not_registered(ingestion: IngestionService, tmp_path: Path) -> None:
    p = tmp_path / "x.txt"
    p.write_text("hi")
    with pytest.raises(UnsupportedFormatError):
        ingestion.ingest(p)
    assert ingestion.count() == 0
