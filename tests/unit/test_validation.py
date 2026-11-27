from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from veritas.core.contracts import DocumentFormat
from veritas.ingestion.validation import (
    CorruptedFileError,
    EmptyFileError,
    FileTooLargeError,
    UnsupportedFormatError,
    ValidationError,
    sha256_file,
    validate_file,
)

MB = 1024 * 1024


@pytest.mark.parametrize("fmt", ["pdf", "docx", "xlsx"])
def test_supported_formats(sample_files: dict[str, Path], fmt: str) -> None:
    vf = validate_file(sample_files[fmt], 10 * MB)
    assert vf.format is DocumentFormat(fmt)
    assert vf.size_bytes == sample_files[fmt].stat().st_size
    assert vf.mime_type.startswith("application/")


def test_extension_case_insensitive(tmp_path: Path, sample_files: dict[str, Path]) -> None:
    p = tmp_path / "REPORT.PDF"
    p.write_bytes(sample_files["pdf"].read_bytes())
    assert validate_file(p, MB).format is DocumentFormat.PDF


@pytest.mark.parametrize("name", ["notes.txt", "deck.pptx", "old.doc", "noext"])
def test_unsupported_extension(tmp_path: Path, name: str) -> None:
    p = tmp_path / name
    p.write_text("hello")
    with pytest.raises(UnsupportedFormatError):
        validate_file(p, MB)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="does not exist"):
        validate_file(tmp_path / "nope.pdf", MB)


@pytest.mark.parametrize("ext", ["pdf", "docx", "xlsx"])
def test_empty_file(tmp_path: Path, ext: str) -> None:
    p = tmp_path / f"empty.{ext}"
    p.touch()
    with pytest.raises(EmptyFileError):
        validate_file(p, MB)


def test_too_large(tmp_path: Path) -> None:
    p = tmp_path / "big.pdf"
    p.write_bytes(b"%PDF-1.4\n" + b"0" * 2048)
    with pytest.raises(FileTooLargeError):
        validate_file(p, 1024)


def test_text_renamed_to_pdf(tmp_path: Path) -> None:
    p = tmp_path / "fake.pdf"
    p.write_text("this is not a pdf")
    with pytest.raises(CorruptedFileError, match="%PDF-"):
        validate_file(p, MB)


@pytest.mark.parametrize("ext", ["docx", "xlsx"])
def test_non_zip_office_file(tmp_path: Path, ext: str) -> None:
    p = tmp_path / f"fake.{ext}"
    p.write_bytes(b"\x00\x01garbage")
    with pytest.raises(CorruptedFileError, match="zip"):
        validate_file(p, MB)


def test_truncated_docx(tmp_path: Path, sample_files: dict[str, Path]) -> None:
    data = sample_files["docx"].read_bytes()
    p = tmp_path / "truncated.docx"
    p.write_bytes(data[: len(data) // 2])
    with pytest.raises(CorruptedFileError):
        validate_file(p, MB)


def test_xlsx_renamed_to_docx(tmp_path: Path, sample_files: dict[str, Path]) -> None:
    p = tmp_path / "wrong.docx"
    p.write_bytes(sample_files["xlsx"].read_bytes())
    with pytest.raises(CorruptedFileError, match="word/document.xml"):
        validate_file(p, MB)


def test_zip_with_corrupt_member(tmp_path: Path) -> None:
    p = tmp_path / "bad.xlsx"
    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("xl/workbook.xml", "<workbook>" + "x" * 500 + "</workbook>")
    raw = bytearray(p.read_bytes())
    i = raw.find(b"xl/workbook.xml") + len("xl/workbook.xml") + 5  # inside compressed data
    raw[i] ^= 0xFF
    p.write_bytes(bytes(raw))
    with pytest.raises(CorruptedFileError):
        validate_file(p, MB)


def test_sha256_matches_hashlib(sample_files: dict[str, Path]) -> None:
    p = sample_files["pdf"]
    assert sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha256_file(p, chunk_size=7) == sha256_file(p)
