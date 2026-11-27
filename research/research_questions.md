# Research questions

Status: **draft**. No experiments have been run. No results exist yet.

## Primary question

**RQ:** To what extent does provenance-preserving, claim-level verification improve contradiction detection and citation reliability, compared with conventional document RAG, on multi-document PDF/DOCX/XLSX corpora?

This is framed as an engineering evaluation. We do not claim it is novel until a literature review supports that.

## Sub-questions and experiments

| ID | Question | Baseline | Treatment | Primary metric |
|---|---|---|---|---|
| E1 | Does retrieving at claim level (not chunk level) improve contradiction detection? | Chunk RAG + LLM judge | Claim-level evidence retrieval | Contradiction F1; citation correctness |
| E2 | Does hybrid retrieval beat lexical retrieval for evidence? | BM25 | Dense + BM25 (RRF) | Recall@5, Recall@10, MRR |
| E3 | Does graph context improve retrieval of related claims? | No graph | Graph-expanded retrieval | Recall@10 of gold evidence |
| E4 | Does a rule + NLI + LLM pipeline beat an LLM alone at verification? | LLM-only | Rule → NLI → LLM cascade | Verdict macro-F1; abstention precision |
| E5 | Does entity resolution reduce missed contradictions? | Exact string match | Resolution pipeline | Pairwise F1; contradiction recall |
| E6 | Does DSPy optimisation improve extraction over manual prompts? | Hand-written prompts | DSPy-optimised | Claim extraction F1 |

## Fairness constraints

* Every arm of an experiment uses the same test split, the same candidate claims and the same available evidence (critical for E4).
* The local model, quantisation and decoding settings are fixed and recorded for each run.
* Scores from different datasets (FEVER, SciFact, HoVer, VERITAS-DocBench) are reported separately and never averaged into a single number.
* Synthetic documents alone are not enough for the final conclusions: results must also be reported on the realistic subset of VERITAS-DocBench.

## Hypotheses (to be tested, not assumed)

* **H1:** Modality- and period-aware comparison reduces false-positive contradictions on the "contextually different" category compared with value-only comparison.
* **H2:** Claim-level citations have higher citation correctness than chunk-level citations, at a similar citation completeness.
* **H3:** Explicit abstention lowers the unsupported-claim rate at an acceptable cost in coverage.

## Open questions

* Which NLI model fits in memory next to Docling and Qwen3 4B on 16 GB?
* How should claims that span table rows (a header plus a value cell) be represented as evidence: one span or several?
* Should the realistic benchmark subset come from public Sri Lankan government or university reports, and what is their licensing?
