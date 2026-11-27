"""File validation before a document is registered.

Checks extension, size and content signature. MIME type is derived from the
verified content, not trusted from the client. No ``python-magic`` dependency:
the three supported formats have simple, well-defined signatures.
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

from veritas.core.contracts import DocumentFormat

MIME_TYPES: dict[DocumentFormat, str] = {
    DocumentFormat.PDF: "application/pdf",
    DocumentFormat.DOCX: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    DocumentFormat.XLSX: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}
# The part every valid OOXML package of this type must contain.
_OOXML_MAIN_PART = {
    DocumentFormat.DOCX: "word/document.xml",
    DocumentFormat.XLSX: "xl/workbook.xml",
}


class ValidationError(ValueError):
    """Base class: the file cannot be ingested."""


class UnsupportedFormatError(ValidationError):
    pass


class EmptyFileError(ValidationError):
    pass


class FileTooLargeError(ValidationError):
    pass


class CorruptedFileError(ValidationError):
    pass


@dataclass(frozen=True)
class ValidatedFile:
    path: Path
    format: DocumentFormat
    mime_type: str
    size_bytes: int
    sha256: str


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def detect_format(path: Path) -> DocumentFormat:
    ext = path.suffix.lower().lstrip(".")
    try:
        return DocumentFormat(ext)
    except ValueError:
        supported = ", ".join(f".{f.value}" for f in DocumentFormat)
        raise UnsupportedFormatError(
            f"unsupported extension {path.suffix!r}; supported: {supported}"
        ) from None


def _check_signature(path: Path, fmt: DocumentFormat) -> None:
    if fmt is DocumentFormat.PDF:
        with path.open("rb") as f:
            head = f.read(1024)
        if b"%PDF-" not in head:
            raise CorruptedFileError(f"{path.name}: missing %PDF- header")
        return

    if not zipfile.is_zipfile(path):
        raise CorruptedFileError(f"{path.name}: not a valid {fmt.value.upper()} (zip) package")
    try:
        with zipfile.ZipFile(path) as z:
            names = set(z.namelist())
            bad = z.testzip()
    except (zipfile.BadZipFile, OSError) as e:
        raise CorruptedFileError(f"{path.name}: {e}") from e
    if bad is not None:
        raise CorruptedFileError(f"{path.name}: corrupted member {bad!r}")
    if _OOXML_MAIN_PART[fmt] not in names:
        raise CorruptedFileError(
            f"{path.name}: missing {_OOXML_MAIN_PART[fmt]}; not a {fmt.value.upper()} file"
        )


def validate_file(path: Path, max_bytes: int) -> ValidatedFile:
    """Validate ``path`` and return its verified metadata, or raise ValidationError."""
    if not path.is_file():
        raise ValidationError(f"{path} does not exist or is not a file")
    fmt = detect_format(path)
    size = path.stat().st_size
    if size == 0:
        raise EmptyFileError(f"{path.name} is empty (0 bytes)")
    if size > max_bytes:
        raise FileTooLargeError(f"{path.name} is {size} bytes; limit is {max_bytes}")
    _check_signature(path, fmt)
    return ValidatedFile(path, fmt, MIME_TYPES[fmt], size, sha256_file(path))
