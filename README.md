# VERITAS AI

### Evidence-Grounded Document Intelligence & Contradiction Detection

**Transform unstructured documents into verifiable knowledge.**

VERITAS AI is an experimental AI engineering project that explores how multiple documents can be transformed into structured knowledge, connected through knowledge graphs, and analyzed to identify contradictions, missing evidence, and unsupported claims.

Unlike conventional Retrieval-Augmented Generation (RAG) systems that primarily focus on retrieving relevant information, VERITAS AI aims to investigate relationships between claims, their supporting evidence, and conflicting information across multiple sources.

The ultimate goal is to generate transparent, evidence-grounded reports in which every factual claim can be traced back to its original document.

> Every fact needs evidence. Every contradiction needs an explanation. Every conclusion must be verifiable.

---

## Project Overview

Organizations often maintain information across multiple PDF reports, Word documents, spreadsheets, and other unstructured sources.

These documents may contain:

- Conflicting numerical values.
- Inconsistent dates and timelines.
- Different descriptions of the same entity.
- Outdated or conflicting information.
- Claims without sufficient supporting evidence.
- Information distributed across multiple documents.

Traditional document retrieval systems can retrieve relevant passages but may struggle to identify whether information across different sources is consistent, contradictory, or insufficiently supported.

VERITAS AI addresses this engineering challenge through document intelligence, structured claim extraction, knowledge graphs, hybrid retrieval, and evidence-based verification.

## Core Capabilities

The following capabilities represent the project's development objectives.

| Capability | Description | Status |
|---|---|---|
| Multi-format ingestion | PDF, DOCX and XLSX validation, hashing, de-duplication, versioning | **Implemented** |
| Document intelligence | Structured text, heading and table extraction (Docling) | **Implemented** |
| Claim extraction | Convert document content into atomic claims | Planned |
| Entity resolution | Identify equivalent entities across documents | Planned |
| Knowledge graph | Connect entities, claims and evidence | Planned |
| Hybrid retrieval | Combine semantic and lexical retrieval | Planned |
| Contradiction detection | Identify conflicting claims | Planned |
| Evidence verification | Verify claims against source documents | Planned |
| Source traceability | Preserve page, sheet and cell references | **Implemented** for parsed spans; claim-level citations planned |
| Report generation | Generate evidence-grounded reports | Planned |

**Note:** Features will be marked as implemented only after development and testing. "Implemented" means code plus passing tests in this repository; it does not imply evaluated quality.

## Getting Started

Requirements: macOS or Linux, Python 3.11+ (3.12 tested), roughly 3 GB of disk for dependencies and Docling models. Ollama is not needed for Milestone 1.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Generate the three sample documents (PDF, DOCX, XLSX)
python scripts/make_sample_documents.py

# Ingest and parse them; prints one inspection report per file
python scripts/ingest.py sample_documents/*

# Tests
pytest -m "not slow"   # unit tests, ~2 s, no models
pytest                 # everything incl. Docling integration tests
                       # (first run downloads models, ~5 min)
```

Configuration comes from `VERITAS_*` environment variables or a `.env` file; see [.env.example](.env.example). Runtime data (SQLite database, stored originals, parsed JSON) goes to `./data/`, which is git-ignored.

**macOS note:** if `import veritas` fails with `ModuleNotFoundError` after an editable install, macOS has set the "hidden" flag on the install's `.pth` file, and Python 3.12 skips hidden `.pth` files. This has been seen inside iCloud-synced `~/Documents`. Run `chflags nohidden .venv/lib/python3.*/site-packages/__editable__*.pth`, or prefix commands with `PYTHONPATH=src`. Pytest is unaffected.

Documentation: [architecture](docs/architecture.md) · [data model](docs/data_model.md) · [ADR-001 technology stack](docs/decisions/ADR-001-technology-stack.md) · [research questions](research/research_questions.md) · [benchmark protocol](research/benchmark_protocol.md)

## System Architecture

Proposed processing pipeline:

```text
          DOCUMENT SOURCES
        PDF | DOCX | XLSX
                 |
                 v
        DOCUMENT INGESTION
                 |
                 v
         DOCUMENT PARSING
              Docling
                 |
                 v
        KNOWLEDGE EXTRACTION
       Entities | Claims | Facts
                 |
                 v
          ENTITY RESOLUTION
                 |
                 v
         KNOWLEDGE GRAPH
                 |
                 v
          HYBRID RETRIEVAL
          Vector + BM25
                 |
                 v
       CONTRADICTION ANALYSIS
                 |
                 v
         EVIDENCE VERIFICATION
                 |
                 v
       EVIDENCE-GROUNDED REPORTS
                 |
                 v
         INTERACTIVE DASHBOARD
```

## Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Document processing | Docling |
| Local LLM inference | Ollama |
| LLM candidates | Qwen3 and other lightweight models |
| Structured validation | Pydantic |
| Embeddings | BGE models |
| Vector database | Qdrant |
| Keyword retrieval | BM25 |
| Knowledge graph | NetworkX |
| Relational database | SQLite |
| ORM | SQLAlchemy |
| LLM pipeline optimization | DSPy |
| Backend | FastAPI |
| Frontend | Streamlit |
| Testing | Pytest |
| Code quality | Ruff |

Technology selections are subject to implementation benchmarks and hardware constraints.

## Research Direction

The primary research question is:

**To what extent does provenance-preserving, claim-level verification improve contradiction detection and citation reliability compared with conventional document RAG?**

Planned research experiments include:

1. Conventional RAG versus claim-level evidence retrieval.
2. BM25 versus dense retrieval versus hybrid retrieval.
3. Graph-free RAG versus knowledge graph-enhanced retrieval.
4. LLM-only contradiction detection versus hybrid verification.
5. Exact entity matching versus semantic entity resolution.
6. Conventional extraction prompts versus DSPy-optimized extraction.

### Evaluation Metrics

The system will be evaluated using:

- Claim extraction precision, recall and F1.
- Entity resolution precision, recall and F1.
- Contradiction detection precision, recall and F1.
- Evidence retrieval Recall@K and MRR.
- Citation completeness and correctness.
- Unsupported claim rate.
- Verification and abstention accuracy.
- Processing latency and memory consumption.

A document-centric benchmark, **VERITAS-DocBench**, is also planned to evaluate cross-document contradictions, contextual differences, missing evidence and source traceability.

## Development Environment

Designed initially for resource-efficient local development.

| Resource | Configuration |
|---|---|
| Development machine | MacBook Air M2 |
| Memory | 16 GB unified memory |
| Operating system | macOS |
| Python | 3.11+ |
| LLM runtime | Ollama |
| Deployment strategy | Local-first, optional cloud inference |
| Paid API dependency | None required |

Cloud inference may be used for optional comparative experiments when a genuinely free allowance is available.

The core application is intended to remain functional without paid APIs.

## Project Structure

```text
veritas-ai/
│
├── src/
│   └── veritas/
│       ├── api/
│       ├── core/
│       ├── ingestion/
│       ├── parsing/
│       ├── extraction/
│       ├── entities/
│       ├── claims/
│       ├── evidence/
│       ├── graph/
│       ├── retrieval/
│       ├── verification/
│       ├── contradictions/
│       ├── reporting/
│       ├── llm/
│       └── evaluation/
│
├── frontend/
├── database/
├── benchmarks/
├── experiments/
├── tests/
├── sample_documents/
├── research/
├── docs/
├── scripts/
│
├── pyproject.toml
├── README.md
└── LICENSE
```

## Development Roadmap

| Phase | Focus |
|---|---|
| Phase 0 | Research, architecture and repository foundation |
| Phase 1 | Document ingestion and parsing |
| Phase 2 | Structured claim and entity extraction |
| Phase 3 | Entity resolution |
| Phase 4 | Knowledge graph construction |
| Phase 5 | Hybrid retrieval |
| Phase 6 | Contradiction detection |
| Phase 7 | Evidence verification |
| Phase 8 | Interactive dashboard and reporting |
| Phase 9 | Research experiments and evaluation |

## Inspiration and References

This project draws engineering inspiration from:

- [Docling](https://github.com/docling-project/docling) — document parsing and document intelligence.
- [RAGFlow](https://github.com/infiniflow/ragflow) — retrieval-augmented generation and document processing.
- [DSPy](https://github.com/stanfordnlp/dspy) — programming and optimization of language model pipelines.
- [Qdrant](https://github.com/qdrant/qdrant) — vector search and hybrid retrieval.
- [FEVER](https://fever.ai/) — fact extraction and verification benchmark.

VERITAS AI is an independent experimental implementation, not an official extension of these projects.

## Project Status

**Current stage: Phase 0 → Milestone 1 (document ingestion and parsing)**

Implemented: core data contracts, document ingestion (validation, SHA-256, de-duplication, versioning, SQLite registration), and Docling parsing with page, structure, sheet and cell provenance. Not yet implemented: claim extraction, entity resolution, knowledge graph, retrieval, contradiction detection, verification, API, dashboard. The system does **not** detect contradictions yet.

This repository is under active development. Experimental results and performance claims will be published only after reproducible evaluation.

## License

MIT License.

See [LICENSE](LICENSE) for details.

---

**VERITAS AI — Making AI-generated knowledge traceable, explainable, and verifiable.**
