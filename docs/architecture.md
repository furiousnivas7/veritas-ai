# Architecture

Status key: **Implemented** = code and tests exist in this repository. **Planned** = design only.

## Principles

1. **Every fact has a source.** Nothing becomes a claim unless it points to a located span of an ingested document.
2. **Local first.** The system runs fully offline on a 16 GB Apple Silicon laptop. Cloud models are optional experiment participants and never a dependency.
3. **The relational store is the source of truth.** The knowledge graph (NetworkX) and the vector index (NumPy exact search by default; Qdrant only as an optional backend, see [ADR-003](decisions/ADR-003-retrieval-and-graph.md)) are derived from it and can be rebuilt.
4. **Abstain instead of guessing.** "Insufficient evidence" and "requires human review" are valid outputs.
5. **Never fabricate provenance.** If the parser does not expose a page or cell, the span records that it is missing; no location is inferred.

## Layers

| Layer | Responsibility | Modules | Status |
|---|---|---|---|
| Interface | Upload, inspect, report | `api/`, `frontend/streamlit_app/` | Planned |
| Application | Workflow orchestration | `ingestion/pipeline.py` | **Implemented** (ingest → parse only) |
| Intelligence | Claim extraction, entity resolution | `extraction/`, `entities/`, `llm/` | **Implemented** (Milestone 2) |
| Intelligence | Verification, contradiction analysis | `verification/`, `contradictions/` | Planned |
| Retrieval | BM25, dense (NumPy exact search), RRF fusion, graph traversal ([ADR-003](decisions/ADR-003-retrieval-and-graph.md)) | `retrieval/`, `graph/` | Planned |
| Persistence | Documents, spans, claims, evidence | `core/db.py` | **Implemented** (`documents`, `document_spans`) |
| Evaluation | Benchmarks, metrics, experiments | `evaluation/`, `benchmarks/`, `experiments/` | Planned |

Shared contracts for all layers live in `core/contracts.py` (**Implemented**).

## Processing lifecycle

```text
 file ─▶ A. Ingestion ─▶ B. Parsing ─▶ C. Claim extraction ─▶ D. Entity resolution
          (implemented)   (implemented)   (implemented)          (implemented, rule-based)
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

### C. Claim and entity extraction — Implemented (`extraction/`, `llm/`)

```text
document_spans ─▶ plan_sources ─▶ batches of [S1..Sn] ─▶ LLMProvider.generate
  (located only)    text units +       (≤ 1800 chars)        (JSON schema or JSON mode,
                    table-row units                           temperature 0, fixed seed)
                                                  │
        invalid JSON / schema ◀── retry ≤ 2 ──────┤ timeout / server error ◀── retry ≤ 2
        unavailable / model missing ─▶ abort run (not retried)
                                                  ▼
               LLMExtraction (strict Pydantic) ─▶ ClaimValidator (deterministic grounding)
                                                  ├─▶ Claim + Evidence (sliced from spans)
                                                  ├─▶ EntityMention
                                                  └─▶ Rejection (reason, payload)  ─▶ audit
                                                  ▼
                      EntityResolver ─▶ save_run (one transaction, span ids re-verified)
```

* **Sources.** Paragraphs, list items, captions and footnotes become text units. Each data row of a table becomes a row unit rendered as `Header = value | …` and backed by its cell spans, so XLSX claims cite the exact cells. Headings and titles are context, not sources. Spans without a location are never sent to the model.
* **Model adapter.** `LLMProvider` (`llm/base.py`) with `OllamaProvider` and `MockProvider`. The model, `think`, constrained-output mode, seed, temperature, context size and timeout all come from settings (`VERITAS_OLLAMA_*`) or `--model`. Nothing else names a model.
* **Prompts** are versioned (`extraction/prompts.py`, `claim_entity_extraction@1.0.0`). Each run stores the prompt id, version and the SHA-256 of prompt + response schema.
* **Validation** is model-independent. See [data_model.md](data_model.md#extraction-pipeline-contracts-milestone-2) for the full rule table.

### D. Entity resolution — Implemented, rule-based (`entities/`)

* **Identity is not the name.** Entity ids are minted from the mention that founded the entity (`ENT-` + SHA-256 of that mention id), and the database allows any number of entities with the same type and normalised name.
* **MERGED** only within the same entity type and the same document: (1) an exact normalised-name or proven-alias match with an entity already mentioned in that document, or (2) an explicit `Long Name (ACR)` definition in that document whose acronym really matches.
* **NEEDS_REVIEW** (never merged): the same name or alias in *another* document (`exact_cross_document`, `alias_definition_cross_document`), reordered words, undefined acronyms, near-identical spelling (difflib ≥ 0.88), aliases claimed only by the model, and names matching several entities in one document (left unassigned).
* Different entity types never merge. Existing entities and the documents they appear in are loaded before each document, so cross-document matches are found and queued for review.
* `VERITAS_ENTITY_RESOLUTION_POLICY=global_name_match` restores name-only merging across documents. It is the explicit baseline for experiment E5, not the default. The policy used is recorded in every run's stats.
* Embedding similarity is not used yet (Milestone 3); when added, it produces candidates only.

### E–F. Planned
* **E. Contradiction analysis:** candidate pairs must share subject (resolved entity) and predicate. Modality, unit and period compatibility are checked *before* values are compared, so that a proposed and an approved budget are recorded as `CONTEXTUALLY_DIFFERENT`, not `CONTRADICTED`.
* **E'. Verification:** rules + NLI + LLM, with explicit abstention.
* **F. Reporting:** reports are generated only from verified records and their evidence; the LLM may phrase them but must not add facts.

## Runtime footprint (measured on the reference machine)

| Operation | Observed | Notes |
|---|---|---|
| First PDF parse | ~287 s | Includes the one-time Docling model download |
| Warm PDF parse (2 pages) | ~11 s | Layout and table models on CPU/MPS |
| DOCX / XLSX parse (samples) | < 0.1 s | No ML models involved |
| Full test suite | ~13 s | 226 tests (1 opt-in live test skipped), Docling models cached |

These are single runs of small sample files, not benchmarks.

## Out of scope for Phase 0

Neo4j, LangChain/LangGraph, multi-agent orchestration, Kubernetes, and any paid API. Any new framework needs an ADR.
