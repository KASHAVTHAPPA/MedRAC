# MedRAC-QA — Mid-Semester Progress Audit

**Project:** A Lightweight LLM + RAG + CAG Framework for Medical Health Question Answering
**Student:** Kashav Thappa (23BCS040) · **Guide:** Prof. (Dr.) Baij Nath Kaushik
**Audit date:** 22 October 2026 · **Build complete by:** 15 November 2026 · **Final evaluation:** 23–27 November 2026

This audit lists every item in the full project scope (from the approved design:
problem formulation, architecture, algorithm, experiments, ablations,
deliverables) and marks it **Done**, **Partial** or **Not started**, with
evidence from the repository. The completion figure is computed from those
marks, not estimated.

## 1. Verification run (22 Oct 2026)

| Check | Result |
|---|---|
| `pytest -q` (unit + API tests; fake LLM, real index, fake Redis) | **58 passed** |
| Knowledge base build (`scripts/build_index.py`) | 16,407 QA pairs → 15,809 unique answers → 27,470 passages, 26.6 MB, 8 min on CPU |
| Local web app (`uvicorn app:app`) | all routes work in the browser (desktop and 500 px mobile): RAG, cache, abstain, emergency |
| Real LLM (Gemini 3.5 Flash-Lite via OpenAI-compatible API) | working; median 4.1 s per RAG answer |
| Vercel bundle estimate (Linux wheels, Python 3.12) | ≈ 184 MB packages + 49 MB index/model ≈ 233 MB |
| Deployed on Vercel | **Live** at https://med-rac.vercel.app (all routes verified: RAG, cache, context rejection, abstain, emergency) |

## 2. Experimental results so far

### 2.1 Cache gate (RQ2, RQ3, ablations A2, A3, A9) — `experiments/eval_cache.py`

Workload: 300 MedQuAD questions in the cache; four probes each (paraphrase,
added clinical context, different intent, most-similar other topic). τ is chosen
on the dev half (rule: best F1 with unsafe reuse ≤ 1%) and reported on the
held-out test half.

| Cache (B3 / B4 / proposed) | τ | Useful hits | Hit precision | Unsafe reuse | F1 |
|---|---|---|---|---|---|
| Exact match | – | 0.7% | 100% | 0.00% | 0.013 |
| Plain semantic | 0.90 | 86.7% | 67.4% | 10.50% | 0.758 |
| **Gated semantic (MedRAC-QA)** | 0.85 | **96.0%** | **93.5%** | **1.67%** | **0.947** |

Wrong reuses on the test half, by probe type (plain → gated): context 34 → **0**,
intent 6 → 1, near topic 23 → 9. The remaining errors are closely related
diseases with different names, such as "pulmonary hypertension" and "pulmonary
arterial hypertension". These are the target of the phase-2 work.

Two corrections were made during the experiment and are recorded here:
(1) MedQuAD lists some diseases under two spellings ("Beta-thalassemia" /
"beta thalassemia"); the ground truth now treats these as the same topic;
(2) the gate was tightened after error analysis on the dev split: subtype codes
such as "1B" count as numbers, and the topic-word overlap threshold went from
0.5 to 0.75.

### 2.2 Retrieval (RQ1, retrieval part) — `experiments/eval_retrieval.py`

TREC 2017 LiveQA-Med: 104 real questions sent to the U.S. National Library of
Medicine, with expert-judged MedQuAD answers. Only 38 questions have a relevant
(grade ≥ 3) answer inside our knowledge base. Most judged answers come from the
three copyrighted MedlinePlus subsets, which we exclude.

| Query form | Hit@1 | Hit@5 | Hit@10 | Recall@10 | MRR@10 |
|---|---|---|---|---|---|
| Original consumer message | 0.32 | 0.58 | 0.71 | 0.51 | 0.42 |
| NLM one-line summary | 0.37 | 0.71 | 0.82 | 0.66 | 0.51 |

Median search time is 3–4 ms. The gap between the raw and summarised question
suggests question rewriting as a phase-2 improvement.

### 2.3 Evidence threshold γ (RQ4, gate part) — `experiments/eval_abstention.py`

| γ | LiveQA answered | MedQuAD answered | Out-of-scope rejected |
|---|---|---|---|
| 0.35 | 98.1% | 100% | 90% |
| **0.40 (chosen)** | **96.2%** | **99.5%** | **96%** |
| 0.45 | 89.4% | 99.5% | 98% |
| 0.50 | 72.1% | 99.5% | 100% |

### 2.4 End-to-end pilot (RQ2) — `experiments/pilot_end_to_end.py`

12 questions × (original, paraphrase, paraphrase + clinical context), plus the
12 originals in LLM-only mode; Gemini 3.5 Flash-Lite.

| Route | Count | Median latency |
|---|---|---|
| RAG (grounded, validated) | 23 | 4.1 s |
| Cache hit | 9 of 12 paraphrases | **5.9 ms** |
| Fallback (answer failed validation, sources shown) | 4 | 3.4 s |
| LLM only (baseline B1) | 12 | 3.3 s |

- All 12 "context" questions bypassed the cache, as intended.
- 85% of grounded answers passed validation, and 98.5% of answer sentences were supported.
- The three paraphrases that missed the cache (e.g. "prognosis" vs "outlook") were safe misses.

This pilot is too small to support claims about answer quality; that is the
phase-2 B1–B6 study.

## 3. Scope checklist

| # | Work item (full scope) | Weight | Status | Score | Evidence |
|---|---|---|---|---|---|
| 1 | Knowledge base: MedQuAD parsing, de-duplication, chunking (180 words ≈ MiniLM's 256-token window), embeddings, versioned index | 10 | Done | 1.0 | `scripts/build_index.py`, `data/index/manifest.json` |
| 2 | Safety screening: emergencies (112/108, Tele-MANAS 14416), never-cache rules, disclaimers | 5 | Partial: rules + unit tests; no dedicated safety evaluation set yet | 0.8 | `medrac_qa/safety.py`, `tests/test_safety.py` |
| 3 | Query analysis: intent, clinical qualifiers, numbers, topic words; compatibility check | 8 | Done | 1.0 | `medrac_qa/query_analysis.py` |
| 4 | CAG semantic cache: 4-check gate, TTL + KB-version freshness, exact/naive baselines, memory/file/shared-Redis backends | 10 | Done | 1.0 | `medrac_qa/cache.py`, `tests/test_cache.py`, `tests/test_redis_cache.py` |
| 5 | RAG retrieval: dense MiniLM retrieval; hybrid BM25 + reranker; biomedical-encoder comparison (A7) | 10 | Partial: dense retrieval done and evaluated | 0.6 | `medrac_qa/knowledge.py`, `results/retrieval_liveqa.json` |
| 6 | Lightweight LLM generation: grounded prompt, provider-agnostic client, LLM-only baseline | 6 | Done | 1.0 | `medrac_qa/llm.py`, `prompts.py` |
| 7 | Evidence gate (γ) and citation validation (semantic + lexical); NLI entailment check | 8 | Partial: NLI not started | 0.75 | `medrac_qa/evidence.py` |
| 8 | Web app + API, deployed on Vercel | 8 | Done | 1.0 | `app.py`, `public/`, https://med-rac.vercel.app |
| 9 | Tests and CI | 5 | Partial: CI not yet run on GitHub | 0.9 | `tests/`, `.github/workflows/ci.yml` |
| 10 | Experiments: retrieval ✔, cache gate ✔, abstention ✔, pilot ✔; B1–B6 answer-quality study ✘, PubMedQA ✘, remaining ablations (A1, A4–A8, A10) ✘, significance tests ✘ | 25 | Partial | 0.4 | `experiments/` |
| 11 | Knowledge-preloading CAG variant (context caching) | 5 | Not started (optional in the design) | 0.0 | — |
| 12 | Synopsis, report, paper draft, demo video | 15 | Partial: synopsis and docs done | 0.2 | `docs/Synopsis.pdf`, `README.md` |

**Weighted completion = 75.5 / 115 ≈ 66%.**

The working demo covers the complete pipeline: safety → CAG → RAG → LLM →
validation → cache. It also has measured results for the cache gate, retrieval
and abstention. The remaining 34% is mostly the full answer-quality evaluation,
retrieval upgrades and the report.

## 4. Deliberate design decisions

1. **ONNX Runtime instead of PyTorch** for MiniLM: same model and outputs, but
   23 MB and CPU-only, so the whole system fits in a serverless function.
2. **Exact NumPy search instead of FAISS**: 27k × 384 vectors take about 1 ms. FAISS
   is mentioned in the design, but it adds nothing at this scale; it can be
   swapped in behind `KnowledgeBase.search` if the corpus grows.
3. **Hosted lightweight LLM (Gemini 3.5 Flash-Lite)** for the demo, because
   Vercel cannot host model weights. The client is OpenAI-compatible, so the
   same code runs a local `qwen2.5:3b` via Ollama (as suggested in the design) by
   changing three environment variables.
4. **Rule-based safety and context checks** rather than an LLM classifier:
   predictable, testable and explainable in a viva. Their limits are measured
   (§2.1) rather than assumed.
5. **Copyright-safe corpus**: the MedlinePlus A.D.A.M./Drugs/Herbs answers are
   excluded even though a judged-answer file containing some of them exists.
6. **Thresholds are tuned on dev data**: τ = 0.85 and γ = 0.40 were chosen by the
   experiment scripts using explicit selection rules, not picked by hand.

## 5. Known limitations

- Embedding similarity and word overlap approximate "the passage supports this
  sentence"; they do not prove it. An NLI check is planned.
- The template-generated paraphrases in the cache workload are cleaner than
  real user wording. LLM-generated or real paraphrases are planned.
- Retrieval on real consumer questions is moderate (Hit@5 = 0.71); questions
  outside MedQuAD's topics are abstained on or fall back to showing sources.
- No clinician has reviewed the answers. Results describe system behaviour,
  not clinical safety.

## 6. Plan to 15 November 2026

| Dates | Work |
|---|---|
| 23 Oct – 31 Oct | B1–B6 study on ~300 questions (LiveQA + held-out MedQuAD): ROUGE-L, semantic similarity, citation precision, latency; PubMedQA accuracy / macro-F1 |
| 1 Nov – 8 Nov | Hybrid BM25 + dense retrieval; biomedical encoder comparison (A7); k sweep (A8); NLI claim check; safety test set |
| 9 Nov – 15 Nov | Knowledge-preloading CAG experiment; paired bootstrap / McNemar tests; freeze results |
| 16 Nov – 22 Nov | Report, paper draft, demo video |
| 23 – 27 Nov | Final evaluation |
