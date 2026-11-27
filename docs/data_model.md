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
| `Entity` | Canonical entity with aliases | **Yes** – `entities`, `entity_aliases` |
| `EntityMention` | One located occurrence of an entity name | **Yes** – `entity_mentions` |
| `EntityMatchCandidate` | Evidence for (or doubt about) a mention–entity identity | **Yes** – `entity_match_candidates` |
| `Claim` | Atomic subject–predicate–object statement | **Yes** – `claims` |
| `Evidence` | Quoted, located text backing a claim | **Yes** – `evidence` |
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
| Extraction run | `RUN-` + 12 hex chars (random) | `RUN-3f9a0c1d2e4b` |
| Claim / evidence / mention | `CLM-`/`EV-`/`MEN-` + 16 hex of SHA-256(run, content) | deterministic within a run |
| Claim fingerprint | SHA-256(document, subject, predicate, modality, type, value, period, spans) | equal across runs and models for the same claim |
| Entity | `ENT-` + 16 hex of SHA-256(founding mention id) | minted once, stored, never recomputed from the name. Entities created before schema v1 keep their old 12-hex ids |
| Span | `<document_id>:<docling self_ref>[:r<row>c<col>]` | `DOC-fe0f08394f35:#/tables/0:r1c1` |

A span id is stable for a given document version and parser version. Because a new file version gets a new `document_id`, spans from different versions never collide.

## Relational tables

Milestone 1: `documents` (unique `sha256`, unique `(filename, version)`) and `document_spans` (FK to documents with cascade delete; page, sheet, cell and element id copied out of the location JSON for querying).

Milestone 2 (additive; Milestone 1 tables unchanged):

| Table | Key constraints | Purpose |
|---|---|---|
| `extraction_runs` | PK `run_id`; FK document; CHECK status ∈ completed/partial/failed | Model, options, prompt id/version/SHA-256, stats: everything needed to reproduce a run |
| `claims` | FK run (cascade), document, `subject_entity_id` (set null); UNIQUE `(run_id, fingerprint)`; CHECK modality, 0 ≤ confidence ≤ 1 | Accepted claims; numeric value, unit and currency denormalised for querying |
| `evidence` | FK claim (cascade), **span** (no cascade), document; UNIQUE `(claim_id, span_id, char_start)`; `position` keeps order | Exact quote, offsets and location of each citation |
| `entities` | PK opaque `entity_id`; **non-unique** index on `(entity_type, normalized_name)` | Distinct entities may share a name |
| `entity_aliases` | UNIQUE `(entity_id, normalized_alias)` | Only aliases proven by a definition in the text |
| `entity_mentions` | FK run, span, entity (set null); UNIQUE `(run_id, span_id, char_start, char_end, entity_type)` | Source references for entities |
| `entity_match_candidates` | UNIQUE `(mention_id, entity_id)`; CHECK status ∈ merged/needs_review/rejected | Why a mention was merged, or why it needs review |
| `rejected_extractions` | FK run (cascade) | Audit of every model candidate that failed grounding, with reason and payload |

### Schema versions and migrations

The schema version is stored in `PRAGMA user_version` and upgraded automatically by `make_engine` (`src/veritas/core/migrations.py`):

| Version | Change |
|---|---|
| 0 | Unversioned: Milestone 1 tables, possibly the first Milestone 2 tables |
| 1 | `entities`: UNIQUE `(entity_type, normalized_name)` removed. `entity_mentions`: unique key gains `char_end` |
| 2 | `evidence.role` (`quote` / `scope` / `header`) added, NOT NULL DEFAULT `quote` (accurate: all earlier evidence came from model quotes). Databases at a version newer than the code are refused |
| 3 | `claims.metadata_json` (list of `MetadataDecision`) added, NOT NULL DEFAULT `[]` (accurate: earlier claims had no enrichment) |

Before changing a non-empty database file, the migration writes a full copy to `<db>.bak-v<old>-<UTC timestamp>` using SQLite's backup API. Constraint changes use SQLite's table-rebuild procedure (foreign keys off → create new table → copy every column → drop → rename → recreate indexes → `PRAGMA foreign_key_check`) in one transaction. Row counts are compared and any mismatch or FK violation rolls back. Ids are never rewritten. `tests/unit/test_migrations.py` runs this against a real v0 database produced by the original Milestone 2 code (`tests/fixtures/legacy_v0_m2.sqlite3`).

**Re-parsing a document that already has claims is blocked** by the foreign keys from `evidence`/`entity_mentions` to `document_spans`. This is intentional: cited spans can never be silently replaced. Re-ingest the changed file as a new version instead.

Planned: `relationships`, `conflicts`, `verifications`, `reports`, `experiments`.

## Evidence roles

Every `Evidence` has a `role`:

| Role | Meaning |
|---|---|
| `quote` | Text the model quoted, located in a data span |
| `scope` | Subject, text value or time expression found in the quote's sentence or table row, added as its own exact slice |
| `header` | Slice of a located table **header** cell that supports a currency, unit, modality or boolean predicate for the value's column. Header text is never data-cell evidence, and an unlocated header can never be support |

## Table headers

**Policy: a row is removed from the data only when it is positively identified as a header. When in doubt, every row is kept as data.** Dropping a real data row silently loses source facts. Missing a header only loses the `Header = value` labels.

Docling's `TableCell.column_header` is not used. On Docling 2.154 it marks row 0 of every DOCX/XLSX table as a header, including headerless tables such as `Project Alpha | pending / Project Beta | 5`, and marks nothing in the sample PDF.

Type contrast alone (text above numbers) is not evidence either. `Project Alpha | pending` over `Project Beta | 5` has the same shape as `Project | Amount` over `Project Alpha | 5`. `detect_header_row()` (`extraction/sources.py`) returns one of three results:

| Result | When | Effect |
|---|---|---|
| `no_header` | single-row table; any row-0 cell is value-like (number, amount, %, date, yes/no); or no column has text over value-like cells only (all-text tables) | all rows are data |
| `header` | first row present is table row 0, **and** every row-0 cell is label-like, **and** at least one column has text over value-like cells only, **and** no row-0 cell repeats a value from its own column | row 0 supplies labels and header evidence |
| `ambiguous` | everything else | same as `no_header`: all rows are data, no labels, no header evidence |

*Label-like* means every word is generic column vocabulary (`HEADER_WORDS`: project, item, amount, budget, cost, date, status, approved, …; connectors such as "of" are ignored; a 4-digit year is allowed), or the cell ends in a parenthesised currency or unit from a closed list (`Amount (LKR)`, `Beneficiaries (persons)`). Names (`Project Alpha`), free text (`pending`), other numbers (`Phase 2`) and non-unit parentheses (`(Phase 2)`) are not label-like.

**Inherent ambiguity.** `Project | Amount` over `Alpha | 5` is classified `header`. Content cannot prove it: a data row could consist of exactly those words. The classification rests on likelihood, not certainty. A data row made only of generic column vocabulary, sitting over a value column, is rare in real reports, while such headers are the norm. Strengthening the heuristic cannot remove this residual risk. Only structural metadata or a person can:

| Option | Status |
|---|---|
| Parser metadata | Not usable. Docling's `column_header` flags every DOCX/XLSX row 0 and no PDF rows. Native signals (DOCX "repeat header row", XLSX table objects or bold header styles) are not exposed by Docling 2.154 |
| User confirmation | Not implemented. A review step over the recorded decisions is the natural place (see below) |
| `VERITAS_TABLE_HEADER_POLICY=never` | Implemented. Every row is data, so no first row is ever withheld, at the cost of all header labels, header evidence, and header-derived currency/unit/modality |

**Nothing is discarded.** A `header` row is withheld only from the *data rows* sent for extraction. Its cells:
* stay in `document_spans`, unchanged;
* are carried on every row unit as `header_spans` (located, citable as `header` evidence);
* are shown to the model as column labels;
* are listed in the run's audit record.

Each run's stats include `table_headers`: the policy, counts per status, and one `decisions` entry per table with `table_ref`, `status`, `reason`, and the `header_span_ids` withheld from the data rows. Every header decision can therefore be audited and, in a later review workflow, overridden.

### Numbers and date-shaped text

Before quantities are read, every **date-shaped** token is masked by its shape, valid or not:
* `YYYY-MM-DD` and `D/M/YYYY` with `-`, `/` or `.`;
* `YYYY-MM` and `M/YYYY` with `-` or `/` only, because `5.2025` and `2025.05` are decimals;
* month-name dates.

So `2025-99-99`, `2025-02-30`, `31/02/2025` and `2025-13-40` contribute no quantities. Masks match whole tokens only: no digit, separator or decimal directly adjacent. Numbers outside the token are therefore untouched ("Paid 2025-99-99 total 5,000" → 5000).

Calendar validity matters only for grounding a claimed date. An invalid date grounds no month or day. Its year still counts as a bare-year mention, and bare years ("in 2025") remain numbers, since they cannot be told apart from counts.

Limitations:
* A real header that uses words outside the vocabulary and has no unit ("Ward | Beneficiaries") is `ambiguous`. Its rows are all kept, but values lose their labels.
* All-text tables never get a header.
* Only row 0 is considered: no multi-row headers and no header columns.
* The vocabulary is English.

## Currency and period enrichment

`extraction/enrichment.py` plus `ClaimValidator` run a **deterministic** enrichment step after all other checks (no model call; `VERITAS_METADATA_ENRICHMENT=on|off`, default `on`). It reads only the claim's approved evidence scope:
* **Prose:** the quoted span and the quote's own sentence.
* **Tables:** the cited row's cells and the value column's own header.

Headings, sheet names, other sentences, rows or columns, and document metadata are never used.

**Currency** (MONEY claims). A currency counts only if it is *attached* to the occurrence of the claim's own value: directly before or after the number with only spaces between ("LKR 7 million", "7 million LKR", "€3m"). For tables it may come from the value cell's text or the value column's header (cited as `header` evidence).

Recognised forms:
* ISO codes, upper case only in the source, so "cad" is not CAD.
* Unambiguous names and symbols ("Sri Lankan rupees", "US$", €, £, ₹).
* `$`, `Rs.`, "rupees", "dollars", "pounds" and `¥` are **ambiguous**. They resolve to nothing unless `VERITAS_CURRENCY_SYMBOL_DEFAULTS` (JSON, e.g. `{"Rs": "LKR"}`) says otherwise. There is no implicit LKR. The effective defaults are recorded in every run's `options_json`, and the same rule applies to model-supplied currencies in both modes.

| Situation | Result |
|---|---|
| model silent, one attached currency | filled, `origin=enrichment` |
| model agrees with the evidence / passes the original quote-or-header rule | kept, `origin=model` |
| model contradicts the attached currency | **neither used**; `conflict` (both recorded), issue `currency_conflict` |
| several currencies on the value, or only an ambiguous symbol | none; `ambiguous`, issue `currency_ambiguous` |
| nothing in the evidence | none; `absent`, issue `currency_absent` |
| NUMBER claim with an attached currency | type unchanged; `not_applied` recorded |

**Period.**
* *Prose:* explicit expressions only. A cue word ("in", "for", "during", "as of", "on") followed by a year, month or day; month ranges ("from January to March 2025"); year ranges ("between 2024 and 2026", "2024–2025"); fiscal years ("FY 2024/25" → 2024..2025).
* *Tables:* a same-row date cell under a header naming a time concept (Date, Year, Period, Month, FY; Start + End give a range). The header is cited.

Rules:
* A date in the quote is preferred to one elsewhere in the sentence. Several distinct dates make the period `ambiguous` (issue `period_ambiguous`).
* Precision is kept: a year stays `2025`.
* 01/02/2025 is reduced to `2025` (`precision_reduced_to_year`). 2025-02-30 gives nothing (`invalid_date_in_evidence`).
* "Q1 2025", "since 2024", "by 2026" are recorded as `unsupported_time_expression`.
* A date that *is* the claim's value is not reused as its period.
* A model period the evidence does not support is never replaced. If the evidence states a different period, it is recorded as `conflict`.
* Headers such as "Budget 2025" and sheet names never supply a period (decision D3).

**Provenance.** Each claim carries `metadata: list[MetadataDecision]`, one per field. Each records `origin` (model / enrichment / none), `status`, `value`, the model's value verbatim, the evidence's candidate value, a `reason`, and `evidence_ids` pointing to the exact located slices that support an accepted value. Run stats count decisions by `field:origin:status`.

**Off.** No enrichment and no decisions, with the previous issue names and confidence penalties. Exception: the approved currency-symbol policy (no implicit `Rs.` → LKR) applies in both modes.

**Known limit.** Text alone cannot tell an event date ("approved in March 2025") from a coverage period ("budget for 2025"). An enriched period records the exact expression it came from.

### Prompt versions

| Version | Status | Change |
|---|---|---|
| `claim_entity_extraction@1.0.0` | kept verbatim (sha256 `0db82588…`, pinned by a test) | original |
| `claim_entity_extraction@1.1.0` | **default** (`VERITAS_EXTRACTION_PROMPT_VERSION`) | "or its context" removed for subject and period. Currency must be written next to the amount in the quotes (or in the cited value cell's header), and null for ambiguous symbols. Periods only from the same quoted sentence/row, never headings or sheet names, at their stated precision |

## Extraction pipeline contracts (Milestone 2)

The model returns `LLMExtraction` (`extraction/schema.py`): candidate claims and entities that reference **source ids** (`S1`, `S2`, …) from the prompt. Candidates are never stored directly. `extraction/validator.py` turns them into `Claim`/`EntityMention` or records a rejection:

| Check | Failing outcome |
|---|---|
| `source_id` was in the batch | reject `unknown_source` |
| every evidence quote is in a located span of that source (case and whitespace tolerant; stored text is sliced from the span) | reject `quote_not_in_source` / `missing_provenance` |
| subject is in the cited evidence: a quote, the quote's own sentence, or the cited cell's table row (then added as its own evidence slice). Headings, sheet names and other sentences never count | reject `subject_not_grounded` |
| text value: same scope as the subject | reject `value_not_grounded` |
| numeric value equals a quantity in the quotes ("5 million" = 5000000). Date expressions are masked first, so the 15 in 2025-01-15 / 15/01/2025 / 15 January 2025 is never a quantity | reject `value_not_grounded` / `invalid_value` |
| date object is a real calendar date, and one date expression in the quotes states it at its precision (year; year + month together; exact day). Ambiguous numeric dates such as 01/02/2025 ground only the year | reject `invalid_value` / `value_not_grounded` |
| modality: the **whole** sentence(s) of the quotes (prose), or the cited row plus the value column's header (tables), is the scope; never other sentences, other columns' headers or headings. Model modality not stated there but another is → reject. Nothing stated → `unknown`. **Several modalities stated** ("proposed, but later approved") → `unknown` + `modality_ambiguous`, whichever fragment was quoted. Exactly the model's → accepted | reject `modality_conflict` / downgrade |
| modality word inside the predicate agrees with the modality field | reject `predicate_modality_conflict` |
| currency code or alias appears in the evidence quotes, or in the column header of the cited cell **that holds the value** (never another column, sentence or heading) | remove currency → `number`, issue `currency_not_grounded` |
| non-currency unit: same rule as currency | remove unit, issue `unit_not_grounded` |
| boolean: a quoted clause contains the subject and the predicate word; polarity from explicit negation; hedged, conditional, interrogative or doubly negated clauses do not count (clauses split at commas and contrastive words; the subject must precede the predicate word). Tables: each cited row is judged separately, only rows containing the subject count, and the `Yes/No` cell must be in that row under a header naming the predicate; conflicting rows are ambiguous | reject `boolean_not_grounded` / `boolean_conflict` |
| period's `time_expression` states every date at its precision and is in the cited evidence (quote, quote's sentence, or cited row; then added as evidence). Headings and sheet names never supply a period | remove period, issue `time_not_grounded` |

Predicates are normalised to snake_case, modality words are moved out (`approved_budget` → `budget` + `approved`), and synonyms map to a small vocabulary (`allocation` → `budget`, `spending` → `expenditure`). Proposed and approved budgets therefore share the predicate `budget` and differ in `modality`, while actual spending is `expenditure`.

`extraction_confidence` = model self-report × penalty per issue (e.g. 0.8 for an ungrounded period, 0.9 after a retry). It is a heuristic, uncalibrated, and entirely separate from `VerificationResult.verification_confidence`.
