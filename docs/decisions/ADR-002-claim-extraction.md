# ADR-002: Claim and entity extraction design

* **Status:** Accepted for Milestone 2 (2026-11-27)
* **Context:** A 4B local model (Qwen3 through Ollama) must turn parsed spans into claims without ever inventing provenance, on a 16 GB laptop, with no paid APIs and no new AI frameworks.

## Decisions

1. **The model proposes; deterministic code decides.** The LLM returns candidates in a strict schema (`LLMExtraction`). A model-independent validator accepts a claim only if its source id, quotes, subject and value are found in the parsed spans. Evidence text is always sliced from the span, never copied from the model. *Why:* small models paraphrase and hallucinate numbers; grounding makes acceptance reproducible and identical across models.
2. **Reject core errors; strip unprovable context.** A wrong quote, value, subject or a contradicting modality rejects the claim. An ungrounded currency or period is removed and recorded as an issue, and an unsupported modality becomes `unknown`. *Why:* never fabricate, but don't discard a correct core fact because of a decorative field.
3. **Modality lives outside the predicate.** `approved_budget` → predicate `budget`, modality `approved`. Proposed and approved budgets share a predicate and differ in modality; actual spending is `expenditure`. *Why:* this lets Milestone 4 tell "contextually different" from "contradicted" with a simple key comparison.
4. **Source ids, not free-text citations.** Sources are labelled `S1..Sn` per batch. Table rows are rendered with their headers, and their cells are the evidence. *Why:* the model cannot invent a page or cell; it can only point to a unit we created.
5. **Bounded retries, typed failures.** At most 2 retries per batch. Schema errors are fed back to the model; timeouts and server errors are retried as-is; an unavailable service or missing model aborts the run. Failed and partial runs are persisted for audit.
6. **Entity merging needs proof.** Merges happen only on exact normalised names or explicit `Name (ACR)` definitions in the text. Everything else (reordered words, undefined acronyms, typos, model-claimed aliases) becomes a `needs_review` candidate. *Why:* a false merge corrupts every later contradiction check, while a missed merge only lowers recall.
   *Amended 2026-11-28:* names do not identify entities. Ids are minted per entity (from the founding mention), the `(type, normalised name)` uniqueness was dropped (schema v1), and exact-name merges are limited to the same document. Cross-document name matches are `needs_review` (`exact_cross_document`). `global_name_match` is kept only as the E5 baseline.
7. **Provider adapter without frameworks.** A small `LLMProvider` protocol with `OllamaProvider` and `MockProvider`. No LangChain, DSPy or instructor. *Why:* the surface we need is one call; tests must run without Ollama.
8. **Versioned, hashed prompts.** `claim_entity_extraction@1.0.0`. Each run stores the prompt version, the SHA-256 of prompt + schema, the model and every generation option (temperature 0, seed 42, `think`, format mode).

## Model findings on the reference machine (Ollama 0.40.18)

| Model | `format=<JSON schema>` | `format="json"` | Notes |
|---|---|---|---|
| `qwen3:4b` | works (4.7 s for a one-sentence probe, `think=False`) | works | Default |
| `qwen3:4b-instruct-2507-q4_K_M` | **hangs** (timeouts at 60 s, 300 s, 1555 s) | works (1.5 s) | Use `VERITAS_OLLAMA_FORMAT=json` |

Both models report the `thinking` capability in `ollama show`, so the capability list cannot choose `think` automatically; it is an explicit setting (default `off`).

## Consequences

* Recall is bounded by the grounding rules. For example, a value stated only in words ("five million") is rejected until number-word parsing is added.
* Confidence is a heuristic (model self-report × issue penalties) and is not calibrated; it must not be read as a probability.
* Re-parsing a document that already has claims is blocked by foreign keys; a changed file must be ingested as a new version.

## Amendment 2026-11-28: deterministic metadata enrichment

The real `qwen3:4b` smoke test returned `currency: null` and no period for every claim, even when its own quote said "LKR 7 million … for 2025". A deterministic enrichment step (no second LLM call) now reads currency and period from the claim's own evidence scope. It records a `MetadataDecision` per field (origin, status, reason, supporting evidence) and never overwrites conflicting model output. Ambiguous currency symbols (`$`, `Rs.`) resolve only through explicit configuration, with no implicit LKR. Prompt 1.1.0 aligns the instructions with this evidence policy; 1.0.0 is kept verbatim. Switch: `VERITAS_METADATA_ENRICHMENT`. Schema v3 adds `claims.metadata_json`. Details: docs/data_model.md.
