# Architecture

Status key: **Implemented** = code and tests exist in this repository. **Planned** = design only.

## Principles

1. **Every fact has a source.** Nothing becomes a claim unless it points to a located span of an ingested document.
2. **Local first.** The system runs fully offline on a 16 GB Apple Silicon laptop. Cloud models are optional experiment participants and never a dependency.
3. **The relational store is the source of truth.** The knowledge graph (NetworkX) and the vector index (Qdrant) are derived from it and can be rebuilt.
4. **Abstain instead of guessing.** "Insufficient evidence" and "requires human review" are valid outputs.
5. **Never fabricate provenance.** If the parser does not expose a page or cell, the span records that it is missing; no location is inferred.

## Layers

| Layer | Responsibility | Modules | Status |
|---|---|---|---|
| Interface | Upload, inspect, report | `api/`, `frontend/streamlit_app/` | Planned |
| Application | Workflow orchestration | `ingestion/pipeline.py` | **Implemented** (ingest → parse only) |
| Intelligence | Claim extraction, entity resolution, verification | `extraction/`, `entities/`, `claims/`, `verification/`, `contradictions/`, `llm/` | Planned |
| Retrieval | BM25, dense, graph traversal | `retrieval/`, `graph/` | Planned |
| Persistence | Documents, spans, claims, evidence | `core/db.py` | **Implemented** (`documents`, `document_spans`) |
| Evaluation | Benchmarks, metrics, experiments | `evaluation/`, `benchmarks/`, `experiments/` | Planned |

Shared contracts for all layers live in `core/contracts.py` (**Implemented**).

## Processing lifecycle

```text
 file ─▶ A. Ingestion ─▶ B. Parsing ─▶ C. Claim extraction ─▶ D. Entity resolution
          (implemented)   (implemented)   (planned)              (planned)
                                              │
       F. Report ◀─ E'. Verification ◀─ E. Contradiction analysis
        (planned)       (planned)            (planned)
```

### A. Ingestion — Implemented (`ingestion/validation.py`, `ingestion/service.py`)

1. Validate: file exists; extension is `.pdf`, `.docx` or `.xlsx`; size is above 0 and at most `VERITAS_MAX_UPLOAD_MB`.
2. Verify the content signature: PDF needs `%PDF-` in the first 1 KB. DOCX/XLSX must be valid zip packages, pass a CRC check (`testzip`), and contain `word/document.xml` or `xl/workbook.xml` respectively. The MIME type is derived from the verified format and is never taken from the client.
3. Compute SHA-256 in streamed 1 MB chunks.
4. De-duplicate: identical content (same hash) returns the existing record, whatever its filename.
5. Version: new content under an existing filename gets `version + 1` and `supersedes` points to the previous document.
6. Store a content-addressed copy (`data/documents/<sha256>.<ext>`) and register the metadata in SQLite.

### B. Parsing — Implemented (`parsing/service.py`, `parsing/provenance.py`)

Docling converts the stored copy. `_SpanBuilder` walks the result in reading order and emits one `DocumentSpan` per heading, paragraph, list item, caption, table and non-empty table cell. Each span gets a `SourceLocation` built only from what Docling exposes:

| Format | Location recorded | Source in Docling 2.154 output |
|---|---|---|
| PDF | 1-based physical page, element id, heading path; table row/col for cells | `item.prov[0].page_no` |
| DOCX | Element id, heading path, table row/col | `item.self_ref`; DOCX items have **no** `prov` |
| XLSX | Sheet name, A1 cell or range, table row/col | Sheet = parent `GroupItem` (label `sheet`). The table's `prov[0].bbox` is in cell units (`l`,`t` = 0-based column/row origin; `r`,`b` exclusive), and cell indices are relative to that origin. |

The behaviour in this table was checked by inspecting Docling's output and is pinned by `tests/integration/test_parsing.py`, including a table that does not start at A1.

Outputs: spans in `document_spans`; a `ParseReport` on the document row; normalised JSON at `data/parsed/<document_id>.json`; and the raw Docling export at `data/parsed/<document_id>.docling.json`, kept so documents can be re-processed without re-parsing.

### C–F. Planned

* **C. Claim extraction:** an LLM (Qwen3 4B through Ollama, called with `think=False` and a JSON-schema `format`) plus rules for table rows produce `Claim` objects. Pydantic validation rejects claims that have no evidence.
* **D. Entity resolution:** exact identifiers → normalised strings → aliases → embedding similarity. `SAME_AS` is created only above a confidence threshold or after human review.
* **E. Contradiction analysis:** candidate pairs must share subject (resolved entity) and predicate. Modality, unit and period compatibility are checked *before* values are compared, so that a proposed and an approved budget are recorded as `CONTEXTUALLY_DIFFERENT`, not `CONTRADICTED`.
* **E'. Verification:** rules + NLI + LLM, with explicit abstention.
* **F. Reporting:** reports are generated only from verified records and their evidence; the LLM may phrase them but must not add facts.

## Runtime footprint (measured on the reference machine)

| Operation | Observed | Notes |
|---|---|---|
| First PDF parse | ~287 s | Includes the one-time Docling model download |
| Warm PDF parse (2 pages) | ~11 s | Layout and table models on CPU/MPS |
| DOCX / XLSX parse (samples) | < 0.1 s | No ML models involved |
| Full test suite | ~56 s | 90 tests, with models cached |

These are single runs of small sample files, not benchmarks.

## Out of scope for Phase 0

Neo4j, LangChain/LangGraph, multi-agent orchestration, Kubernetes, and any paid API. Any new framework needs an ADR.
