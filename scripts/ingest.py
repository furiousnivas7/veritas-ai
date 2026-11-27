"""Ingest and parse documents, printing one inspection report (JSON) per file.

Usage:
    python scripts/ingest.py sample_documents/*.pdf sample_documents/*.docx
    python scripts/ingest.py --reparse path/to/file.xlsx
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from veritas.core.config import get_settings
from veritas.core.logging import configure_logging
from veritas.ingestion.pipeline import IngestionPipeline
from veritas.ingestion.validation import ValidationError


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--reparse", action="store_true", help="re-parse files already ingested")
    args = ap.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    pipeline = IngestionPipeline(settings)

    failures = 0
    for f in args.files:
        try:
            res = pipeline.run(f, reparse_duplicates=args.reparse)
        except ValidationError as e:
            failures += 1
            print(json.dumps({"file": str(f), "status": "rejected", "error": str(e)}))
            continue
        out: dict[str, object] = {
            "file": str(f),
            "document_id": res.ingest.document.document_id,
            "version": res.ingest.document.version,
            "duplicate": res.ingest.duplicate,
        }
        if res.report:
            out |= res.report.model_dump(mode="json", exclude={"document_id", "filename"})
            out["export"] = str(res.export_path)
            failures += res.report.status.value == "failed"
        print(json.dumps(out, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
