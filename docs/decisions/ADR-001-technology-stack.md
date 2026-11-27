# ADR-001: Initial technology stack

* **Status:** Accepted for Milestone 1 (2026-11-27)
* **Context:** One developer, 15 h/week, MacBook Air M2 with 16 GB, zero paid API budget, 12-week initial window.

## Decision

| Concern | Choice | Why | Status |
|---|---|---|---|
| Language | Python 3.12 (project requires ≥ 3.11) | Docling / ML ecosystem; 3.12 is the newest version with a smooth torch + Docling install here | In use |
| Parsing | Docling 2.154 | PDF + DOCX + XLSX in one structured model, with page provenance (PDF) and sheet/cell-unit bboxes (XLSX) | In use |
| Contracts | Pydantic v2 | Strict validation of parser and LLM output; JSON Schema for constrained decoding | In use |
| Config | pydantic-settings | Typed env vars (`VERITAS_*`), `.env` support | In use |
| Source of truth | SQLite + SQLAlchemy 2.0 | Zero-ops, file-based, transactional | In use |
| Logging | stdlib `logging` + JSON formatter | Structured logs without a new dependency | In use |
| Tests / lint / types | pytest, Ruff, mypy (strict on `src/`) | Standard, fast | In use |
| Local LLM | Ollama + `qwen3:4b` (2.5 GB, already pulled) | Fits in memory alongside Docling; Phi-4-mini as the comparison candidate | Milestone 2 |
| Embeddings | `BAAI/bge-small-en-v1.5` | Small English baseline; multilingual (e.g. BGE-M3) evaluated separately | Milestone 3 |
| Vector search | Qdrant (local mode) | Dense + sparse retrieval with RRF fusion | Milestone 3 |
| Lexical baseline | BM25 (`rank-bm25`) | Testable baseline independent of Qdrant | Milestone 3 |
| Graph | NetworkX | In-memory graph built from SQLite; no graph DB needed at this scale | Milestone 3 |
| API / UI | FastAPI / Streamlit | Thin interfaces over the services | Milestone 5 |
| Prompt optimisation | DSPy | Only after a measured manual-prompt baseline exists (experiment E6) | Milestone 6 |

## Consequences

* **Only Milestone 1 dependencies are installed** (`pyproject.toml`). Each later dependency is added in the milestone that uses it, which keeps the install small and easy to debug.
* **Torch comes in transitively through Docling** (~2 GB environment). Sentence-transformers will reuse it.
* **No `python-magic`.** The three supported formats are recognised by their signatures and OOXML main parts, which avoids a native libmagic dependency.
* **No Alembic yet.** The schema is created with `create_all`. Migrations start the first time the schema has to change while data already exists.
* **Qwen3 thinking mode.** `qwen3:4b` reasons in "thinking" mode by default, which corrupts JSON output. The provider must pass `think=False` or use `qwen3:4b-instruct-2507`.

## Rejected for Phase 0

Neo4j (NetworkX suffices at this scale), LangChain/LangGraph (plain modules are enough and easier to evaluate), multi-agent orchestration, Kubernetes, and paid APIs.

## Revisit when

* The corpus outgrows in-memory NetworkX (around 10^6 edges), or
* Docling's PDF latency (~11 s for a warm 2-page PDF) becomes the bottleneck for the benchmark, or
* Multilingual (Tamil/Sinhala) support is in scope.
