"""Milestone 1 pipeline: ingest -> parse -> persist -> export. No LLM, no network."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from veritas.core.config import Settings
from veritas.core.db import make_engine, make_session_factory
from veritas.ingestion.service import IngestionService, IngestResult
from veritas.parsing.repository import save_parse
from veritas.parsing.service import DocumentParser, ParseReport


@dataclass(frozen=True)
class PipelineResult:
    ingest: IngestResult
    report: ParseReport | None  # None when a duplicate was skipped
    export_path: Path | None


class IngestionPipeline:
    def __init__(self, settings: Settings, parser: DocumentParser | None = None) -> None:
        self.settings = settings
        self.sessions = make_session_factory(make_engine(settings.resolved_database_url))
        self.ingestion = IngestionService(
            self.sessions, settings.documents_dir, settings.max_upload_mb * 1024 * 1024
        )
        self.parser = parser or DocumentParser(pdf_ocr=settings.pdf_ocr)

    def run(self, path: Path, *, reparse_duplicates: bool = False) -> PipelineResult:
        """Raises ``veritas.ingestion.validation.ValidationError`` for rejected files."""
        ing = self.ingestion.ingest(path)
        if ing.duplicate and not reparse_duplicates:
            return PipelineResult(ing, None, None)
        parsed = self.parser.parse(ing.document, ing.storage_path)
        save_parse(self.sessions, parsed)
        out = self.parser.export(parsed, self.settings.parsed_dir)
        return PipelineResult(ing, parsed.report, out)
