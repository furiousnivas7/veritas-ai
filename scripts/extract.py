"""Extract claims and entities from parsed documents and print them with provenance.

Usage:
    python scripts/extract.py                       # all parsed documents
    python scripts/extract.py DOC-1234abcd5678      # specific documents
    python scripts/extract.py --model qwen3:4b-instruct-2507-q4_K_M
"""

from __future__ import annotations

import argparse
import json
import sys

from veritas.core.config import get_settings
from veritas.core.contracts import Claim, Quantity
from veritas.core.logging import configure_logging
from veritas.extraction.pipeline import run_extraction


def _value(c: Claim) -> str:
    o = c.object
    if isinstance(o, Quantity):
        return " ".join(x for x in (o.currency, format(o.value, "f"), o.unit) if x)
    return str(o)


def _where(c: Claim) -> list[str]:
    out = []
    for e in c.evidence:
        loc = e.location
        if loc.page_number:
            place = f"p.{loc.page_number}"
        elif loc.sheet_name:
            place = f"{loc.sheet_name}!{loc.cell_range}"
        else:
            place = " > ".join(loc.heading_path) or "body"
        out.append(f'{e.document_id} {loc.element_id} [{place}] "{e.quote}"')
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("document_ids", nargs="*")
    ap.add_argument("--model", help="override VERITAS_OLLAMA_MODEL")
    ap.add_argument("--json", action="store_true", help="print machine-readable output")
    args = ap.parse_args(argv)

    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    runs = run_extraction(settings, args.document_ids or None, model=args.model)

    for run in runs:
        if args.json:
            print(
                json.dumps(
                    {
                        "run_id": run.run_id,
                        "document_id": run.document_id,
                        "status": run.status,
                        "model": run.model,
                        "prompt": f"{run.prompt_id}@{run.prompt_version}",
                        "error": run.error,
                        "stats": run.stats(),
                        "claims": [
                            ec.claim.model_dump(mode="json") | {"issues": ec.issues}
                            for ec in run.claims
                        ],
                        "rejections": [vars(r) for r in run.rejections],
                    },
                    indent=2,
                    default=str,
                )
            )
            continue
        print(
            f"\n== {run.document_id}  {run.status}  model={run.model}  "
            f"prompt={run.prompt_id}@{run.prompt_version}  run={run.run_id}"
        )
        if run.error:
            print(f"   error: {run.error}")
        for ec in run.claims:
            c = ec.claim
            period = f" [{c.valid_from}..{c.valid_to}]" if c.valid_from or c.valid_to else ""
            print(
                f"  {c.subject} | {c.predicate} | {_value(c)} | {c.modality.value}{period}"
                f"  conf={c.extraction_confidence}" + (f" issues={ec.issues}" if ec.issues else "")
            )
            for w in _where(c):
                print(f"      <- {w}")
        for r in run.rejections:
            print(f"  REJECTED {r.kind} {r.source_id}: {r.reason}: {r.detail}")
        print(f"   stats: {json.dumps(run.stats())}")
    return 0 if all(r.status != "failed" for r in runs) else 1


if __name__ == "__main__":
    sys.exit(main())
