# ADR-003: Hybrid retrieval and knowledge graph (Milestone 3)

* **Status:** Accepted, from Milestone 3 Blueprint v4.2 (2026-11-28). P0 recorded here; implementation starts at P1.
* **Context:** M1 (ingestion, provenance-preserving parsing) and M2 (claim/entity extraction, deterministic grounding, metadata enrichment) are complete at `e8cbb71`. M3 must retrieve claims and evidence across documents with full provenance. It has to run locally on a MacBook Air M2 (16 GB) at zero cloud/API cost, without contradiction detection or claim verification.
* **Supersedes:** the Milestone 3 rows of ADR-001 that planned Qdrant dense + sparse retrieval with Qdrant-native fusion (see D6 and D8).

## Decisions

| # | Decision |
|---|---|
| D1 | **SQLite is the only source of truth.** Vector indexes, index artifacts and the NetworkX graph are derived and can always be rebuilt from SQLite. |
| D2 | **Active runs:** M2 semantics are preserved (latest `completed` or `partial` run by `started_at`; a newer partial run replaces an older completed one). For identical `started_at`, `run_id DESC` is added as a tie-breaker. This is the only M1/M2 code change: `latest_run_id` delegates to a shared `active_run_ids()`. It narrows previously undefined tie behaviour, and is tested separately (equivalence without ties; determinism with ties). |
| D3 | **Document versions:** the previous version stays active until a newer parsed version has an eligible active run. |
| D4 | **Retrieval items:** active claims, located text spans, and table rows. A table row has a derived ID, no location of its own, and hydrates to its real cell spans (and header cell spans as context). No synthetic source locations. |
| D5 | Text and table-row items come from M2 `plan_sources()` **unchanged**, so there is one rendering and header-decision logic. |
| D6 | **NumPy** exact cosine search is the default vector backend. Qdrant (`qdrant-client==1.20.0`, extra `[qdrant]`) is optional, not installed, and not needed for core completion. |
| D7 | Exact search **masked by an eligibility set computed from SQLite** before the top-N cut, so structural filters never starve candidate depth. The optional Qdrant backend uses non-authoritative payload hints plus oversampling and SQLite verification. |
| D8 | **RRF over ranks only**, implemented in VERITAS: `Σ w_r / (k + rank_r)`, `k = 60`, weights recorded, deterministic tie-breaks. Qdrant-native fusion is not used. |
| D9 | **Entity review** is event-sourced per entity pair (OPEN / DEFERRED / ACCEPTED / REJECTED). Only human events change state; automated observations never do. A reviewer name is required and stored as an **audit label, not authentication**. |
| D10 | **Graph ranking** is deterministic and score-free (an ordering fed to RRF by rank). Ties are fully broken, and traversal paths are recorded on each hit. |
| D11 | **English-only scope.** BGE-small-en-v1.5 is the baseline. multilingual-e5-small is an optional English-only comparison and is not fetched in P0. No Tamil, Sinhala or cross-lingual capability is claimed. Known blocker: M2 `normalize_name` breaks Tamil and Sinhala words at combining marks (separate task, R5). |
| D12 | **Synthetic benchmark first** (VERITAS-RetrievalBench v0); a licensed realistic track comes later and is reported separately. |
| D13 | `previous_version_same_name` review candidates are deferred and outside M3 acceptance. |
| D14 | No M1/M2 contract changes beyond D2; no contradiction detection or claim verification in M3. |
| D15 | **Two fingerprints.** `index_fingerprint` covers everything that determines stored vectors and their interpretation; it drives index identity, selection and rebuilds, with one `ready` index per fingerprint and several fingerprints allowed to coexist. `retrieval_fingerprint` = `index_fingerprint` plus query-side, lexical, fusion and graph settings; it is recorded on every result and benchmark run. |
| D16 | **Freshness:** dense and hybrid search run only when the index state exactly matches the current corpus snapshot on `(item_kind, item_id, document_id, run_id, content_sha256)`. Otherwise `StaleIndexError`, or an explicit `sync="auto"`. BM25 is built from the current snapshot and stays available. Results from an incomplete index are never reported as complete. |
| D17 | `hybrid_graph` requires `CLAIM` in the query's kinds; graph candidates are intersected with the eligibility set, so graph mode can't bypass filters. |
| D18 | **Lineage validation, failing closed.** M1 guarantees linear per-filename chains only through `IngestionService.ingest()`. Branches, cross-filename links, gaps, cycles and missing parents exclude the whole lineage and are reported. |
| D19 | **Index artifact protocol:** build in `.tmp-<id>/` → verify checksums, count and dimension → atomic rename → in one SQLite `BEGIN IMMEDIATE` transaction, re-check the corpus against the build snapshot, then switch `ready` and retire the predecessor. A changed corpus aborts the switch and keeps the previous index. One previous verified index is kept per fingerprint (R8). |
| D20 | **The review state machine is enforced in the database:** an immutable transitions table, a BEFORE INSERT trigger (ordered pair, both entities exist, same type, an observation exists, `pair_seq` continuity, `from_state`, an allowed transition, a note for reverse/reopen, a non-empty reviewer), append-only triggers, `UNIQUE(entity_a, entity_b, pair_seq)` and `BEGIN IMMEDIATE`. |
| D21 | **Corpus snapshot rebuilt per query** on one connection inside one read transaction (correctness first; `PRAGMA data_version` and `total_changes()` are connection-relative). A trigger-based revision counter would touch M1/M2 tables and needs separate approval. |
| D22 | **Fixed execution profile** for the core benchmark: CPU, fixed embedding batch size, fixed intra-op threads, inter-op threads = 1, `torch.use_deterministic_algorithms(True)`, float32 matmul precision `highest`, BLAS threads = 1. Indexing execution settings are part of `index_fingerprint`; query and scoring settings are part of `retrieval_fingerprint`. Different profiles are never claimed equivalent. |

## Resolved items

| Id | Resolution |
|---|---|
| R1 | Benchmark split: **4 development / 6 test families** (decided in Blueprint v4, carried unchanged through v4.2). |
| R2 | multilingual-e5-small is optional (outside core completion); not fetched in P0. |
| R3 | BGE query instruction **off** for the baseline; **on** only as a separate experiment with its own `retrieval_fingerprint`. |
| R4 | Heading paths are included in text-span rendering as **retrieval context, never evidence**. |
| R5 | Multilingual normalisation (M2 `normalize_name`) is a separate task outside M3. |
| R6 | **Superseded by the Blueprint v4 §9 abstention evaluation**, carried unchanged through v4.1 and v4.2. The v3 question ("top dense cosine only, or a combined signal?") was replaced by: three signals reported separately — `s_bm25` (top-1 BM25 score over the eligibility set), `s_dense` (top-1 cosine over the eligibility set, per embedding profile) and optional `s_comb = (z_bm25 + z_dense) / 2` with z-statistics from development queries only, then frozen. AUROC (answerable vs. unanswerable) per signal with family-level bootstrap intervals. Thresholds per signal chosen on development data only (maximise balanced accuracy; ties go to the higher threshold) and applied to test unchanged; the runner refuses to fit anything on the test split. From v4 on, R6 no longer appears in the open-decision lists; v4.2's "R2–R8 resolved" did not list it separately. |
| R7 | Abandoned index-build timeout: 30 minutes, configurable and recorded. |
| R8 | Keep one previous verified index per fingerprint for rollback. |

## P0 record

**Branch:** `feature/m3-retrieval`, created from `e8cbb71`.

**Dependencies** (`pyproject.toml`, `requirements.lock`):

| Package | Pin | Status | Licence |
|---|---|---|---|
| rank-bm25 | 0.2.2 | required; installed in P0 (only new package; depends on numpy only) | Apache-2.0 |
| networkx | 3.7 | required; already installed, now declared | BSD-3-Clause |
| psutil | 7.2.2 | required; already installed, now declared | BSD-3-Clause |
| qdrant-client | 1.20.0 | optional extra `[qdrant]`; **not installed** | Apache-2.0 |

**Model** (`config/models/bge-small-en-v1.5.json`, fetched by `scripts/fetch_models.py`):

* `BAAI/bge-small-en-v1.5` at revision `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`; licence MIT.
* Ten files (134,411,157 bytes). Weights are `model.safetensors` only: no `pytorch_model.bin`, no ONNX. Every file is verified against the hub's LFS SHA-256 or git blob SHA-1, and its local SHA-256 is recorded.
* From the downloaded configuration: BERT, 12 layers, hidden size 384, 12 heads, max position 512, `BertTokenizer` with lower-casing, `model_max_length` 512, CLS pooling, `Normalize` module (L2), float32.
* Query instruction (recorded, off for the baseline per R3): `Represent this sentence for searching relevant passages: `. No passage prefix.

**Readiness:** `scripts/preflight_m3.py` performs read-only checks:
* Python, Git state;
* iCloud paths and offloaded files;
* disk (download + 200 MiB + 1 GiB margin), memory;
* exact dependency pins, core libraries, optional Qdrant (informational);
* offline model checksum verification;
* SQLite version, partial unique indexes and `RAISE(ABORT)` triggers;
* torch deterministic APIs, threads, MPS, NumPy BLAS;
* data-directory writability (checked without writing), HF cache, Ollama (informational);
* the process environment is unchanged afterwards.

Read-only guarantees: Git runs with `--no-optional-locks`; no bytecode is written; SQLite probes use an in-memory database; the model check parses and validates the manifest once (exact file list and order, no duplicates, sizes and totals, checksum and hub-ID formats) and reports any manifest or cache problem as FAIL. Memory only warns (total below the 16 GiB reference, or under 2 GiB available); the download estimate comes from the pinned spec. `fetch_models.py` fails closed when the hub gives no size or no content identity for a file, and replaces the manifest atomically.

## Open assumptions (resolved by tests or measurements, not decisions)

| Id | Assumption | Where it is resolved |
|---|---|---|
| A1 | A renamed file starts a new lineage (M1 keys versions on filename). | documented M1 behaviour; lineage tests (P1) |
| A2' | A per-query snapshot rebuild is fast enough (target p95 ≤ 500 ms on the benchmark corpus). P0 probe: about 0.15 ms per item on a 20-item corpus; a linear estimate for a few thousand items may exceed the target. | P1 measurement |
| A3 | Each benchmark split uses its own temporary database and data directory. | P9 runner |
| A4' | Trigger and partial-unique-index behaviour on SQLite 3.51.3. P0 prototype probe passed, including with `foreign_keys=OFF`. | full raw-SQL matrix (P2) |
| A5 | `F_FULLFSYNC`/`fsync` plus atomic rename on APFS are durable enough for D19. `F_FULLFSYNC` is available. | crash tests (P5) |
| A6 | Rollback-journal locking (an open read transaction blocks a writer's commit; P0 confirmed) is acceptable. | P1/P11 measurement |
| A8 | Artifact files can be hashed offline after fetch. **Confirmed in P0** (`fetch_models.py --verify`). | done |
| A7 | *Withdrawn in Blueprint v4.2* (equivalence across batch sizes and thread counts); replaced by D22 and A9. | — |
| A9 | Repeat runs under the fixed profile are bit-identical. P0 probe with a random-weight model of BGE-small's shape: same profile repeats bit-identical; a different batch size is **not** bit-identical (max difference 7.5e-8), which confirms D22. | P4/P12 with the real weights |

## Known limitations (deferred, not fixed in P0)

* **L7, lock file:** `requirements.lock` records exact versions from one tested environment (Python 3.12.11, macOS arm64) without hashes. It is not a cross-platform or hash-verified lock, and the preflight does not check for drift between the installed packages and the lock.
* **L8, query instruction:** the BGE query instruction text (including its trailing space) is recorded from the model card; the card (`README.md`) is deliberately not fetched, so the text can't be re-checked offline. It doesn't affect the baseline, where the instruction is off (R3).

## Consequences

* M3 adds tables only (migration v4); M1–M3 tables are unchanged. The production database is migrated only after explicit approval, with a backup.
* Results always carry both fingerprints, the index IDs and the corpus signature, so any result can be traced to the exact model, rendering and execution configuration.
* Freshness errors are explicit. Callers decide whether to sync (embedding work is never hidden).
* The English-only limitation, and the M2 normalisation blocker for Tamil and Sinhala, must appear in the README and in benchmark reports.
