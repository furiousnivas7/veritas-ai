# Data model

All contracts are Pydantic v2 models in [`src/veritas/core/contracts.py`](../src/veritas/core/contracts.py). They reject unknown fields (`extra="forbid"`) because LLM output will be validated against them.

## Entities and relations

```text
Document 1──* DocumentSpan 1──* Evidence *──1 Claim *──1 Entity (subject)
                                                │
                       Claim *──* Claim  via Conflict
                       Claim 1──* VerificationResult
                       Entity/Claim/Span ──* Relationship (graph edges)
```

| Model | Purpose | Persisted (SQLite) |
|---|---|---|
| `Document` | One registered file version | **Yes** – `documents` |
| `DocumentSpan` | One parsed element with its location | **Yes** – `document_spans` |
| `SourceLocation` | Citable location, validated per format | Yes – inside `document_spans.location_json` |
| `Entity` | Canonical entity with aliases | Planned (Milestone 2) |
| `Claim` | Atomic subject–predicate–object statement | Planned (Milestone 2) |
| `Evidence` | Quoted, located text backing a claim | Planned (Milestone 2) |
| `Relationship` | Typed graph edge | Planned (Milestone 3) |
| `Conflict` | Outcome of comparing two claims | Planned (Milestone 4) |
| `VerificationResult` | Outcome of checking one claim against evidence | Planned (Milestone 5) |

## Invariants enforced by validation

| Rule | Where |
|---|---|
| A claim has ≥ 1 evidence, with unique evidence ids | `Claim.evidence` (`min_length=1`), `_check_claim` |
| One claim may cite several documents and formats | `Claim.evidence: list[Evidence]` |
| PDF location requires `page_number` | `SourceLocation._check_format_fields` |
| XLSX location requires `sheet_name` + `cell_range` (A1 or A1:C10) | same |
| Page fields are PDF-only; sheet/cell fields are XLSX-only | same |
| DOCX location = `element_id` + `heading_path` (+ table row/col) | same |
| Table coordinates come in pairs and need `table_ref` | same |
| A span has a location **or** a `missing_provenance` reason, never both or neither | `DocumentSpan._location_or_reason` |
| Evidence cannot be built from an unlocated span, or from a quote that is not literally in the span | `Evidence.from_span` |
| Evidence offsets must match the quote length | `Evidence._check_offsets` |
| Document versioning: `version ≥ 1`, `supersedes` links to the previous version | `Document`, `IngestionService` |
| Extraction and verification confidence are separate fields | `Claim.extraction_confidence`, `VerificationResult.verification_confidence` |
| Dates are `YYYY`, `YYYY-MM` or `YYYY-MM-DD`; `valid_to ≥ valid_from` at the coarser precision | `Claim` |

## Claims: modality and negation

`Claim.modality` ∈ {`proposed`, `approved`, `actual`, `forecast`, `unknown`}. Without it, "proposed LKR 7M" and "approved LKR 5M" would be reported as a contradiction. `Claim.negated` represents "X did not happen", which logical contradictions need.

`Claim.object` is a `Quantity` (Decimal value + unit), a `bool` or a `str`. Decimal is used so money values do not suffer float rounding.

## Verdicts and conflict types

Two vocabularies, kept separate:

**`Verdict`**: what the system concluded.

| Verdict | Applies to | Meaning | Extra requirement |
|---|---|---|---|
| `supported` | one claim | Evidence entails the claim | `supporting_evidence_ids` non-empty |
| `refuted` | one claim | Evidence states the opposite | `refuting_evidence_ids` non-empty |
| `insufficient_evidence` | one claim | Nothing found either way (**abstention**) | `abstention_reason` |
| `requires_human_review` | either | Cannot decide confidently (**abstention**) | `abstention_reason` (claims) |
| `contradicted` | claim pair | Same subject, predicate, modality and period; both cannot be true | `conflict_type` |
| `contextually_different` | claim pair | Values differ but contexts differ | no `conflict_type` |

**REFUTED vs CONTRADICTED:** *refuted* compares a claim with **evidence**; *contradicted* compares two **claims**.

**`ConflictType`** (only for `contradicted`): `numerical`, `temporal`, `categorical`, `entity`, `logical`. "Contextual" and "missing evidence" are deliberately *not* conflict types. They are the verdicts `contextually_different` and `insufficient_evidence`.

Every `Conflict` requires a non-empty `explanation` and a `method` (e.g. `rule:numeric`, `nli`, `llm:qwen3:4b`).

## Identifiers

| Object | Format | Example |
|---|---|---|
| Document | `DOC-` + 12 hex chars (random) | `DOC-fe0f08394f35` |
| Span | `<document_id>:<docling self_ref>[:r<row>c<col>]` | `DOC-fe0f08394f35:#/tables/0:r1c1` |

A span id is stable for a given document version and parser version. Because a new file version gets a new `document_id`, spans from different versions never collide.

## Relational tables

Implemented: `documents` (unique `sha256`, unique `(filename, version)`) and `document_spans` (FK to documents with cascade delete; page, sheet, cell and element id copied out of the location JSON for querying).

Planned, from the Phase 0 specification: `entities`, `entity_aliases`, `claims`, `evidence`, `relationships`, `conflicts`, `verifications`, `reports`, `experiments`.
