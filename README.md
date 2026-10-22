# MedRAC-QA

**A Lightweight LLM + RAG + CAG Framework for Medical Health Question Answering**

MedRAC-QA answers health questions in plain language using **only trusted NIH
sources**, shows a citation for every claim, and says *"I don't know"* instead of
guessing. Repeated or reworded questions are answered from a **validated semantic
cache** in a few milliseconds. The cache refuses to reuse an answer when the
clinical context differs, for example "in children" or "during pregnancy".

| | |
|---|---|
| **Author** | Kashav Thappa · Entry No. 23BCS040 |
| **Guide** | Prof. (Dr.) Baij Nath Kaushik, Professor, School of Computer Science & Engineering |
| **Course** | B.Tech CSE, Project Work (7th semester, 4 credits) |
| **Institute** | Shri Mata Vaishno Devi University, Katra (J&K) · Session 2026-27 (Odd) |
| **Licence** | [MIT](LICENSE) (data and model keep their own licences, see below) |

> **Not medical advice.** This is an educational research prototype. It gives
> general health information, not diagnosis or treatment.

---

## How it works

```
 question
    │
    ▼
 1. Safety screen ──── emergency? ──► fixed guidance: call 112 / 108   (no AI involved)
    │
    ▼
 2. Embed question (all-MiniLM-L6-v2, ONNX, CPU)
    │
    ▼
 3. CAG: semantic cache ── validated hit? ──► reuse answer + citations   (~5 ms)
    │      gate = similarity ≥ τ  AND  fresh  AND  same intent / clinical context / topic
    │             AND  evidence still exists
    ▼ miss
 4. RAG: top-5 of 27,470 NIH passages (MedQuAD) by cosine similarity
    │
    ▼
 5. Evidence gate ── best score < γ? ──► abstain ("not in my sources")
    │
    ▼
 6. Lightweight LLM writes a short answer citing [1]..[5]
    │
    ▼
 7. Citation validation ── unsupported? ──► fallback: show the sources only
    │
    ▼
 8. Answer + citations; stored in the cache if safe to reuse
```

**Why the cache gate matters.** "What are the symptoms of diabetes?" and "…of
diabetes *in children*?" are over 90% similar as embeddings, but they need
different answers. A plain semantic cache would reuse the wrong one. The gate
checks the question's **intent** (symptoms, treatment, causes …), its
**clinical qualifiers** (children, pregnancy, elderly, negation), **type numbers**
(type 1 vs type 2) and **topic words** before reusing anything. Dosage, personal
and time-sensitive questions are never cached at all.

## Results so far (mid-semester)

All numbers come from the scripts in [`experiments/`](experiments/). The raw
JSON is in [`experiments/results/`](experiments/results/).

**Cache safety** (300 cached questions, 1,200 probes; τ chosen on the dev half,
reported on the test half):

| Cache | Useful hits | Hit precision | Unsafe reuse |
|---|---|---|---|
| Exact match | 0.7% | 100% | 0.00% |
| Plain semantic (τ = 0.90) | 86.7% | 67.4% | 10.50% |
| **MedRAC-QA gated (τ = 0.85)** | **96.0%** | **93.5%** | **1.67%** |

With the gate, the plain cache's 34 wrong reuses for "children / pregnancy /
older adults" questions dropped to **0**.

**Retrieval** on 38 real consumer questions (TREC 2017 LiveQA-Med) whose
expert-judged answers are in the knowledge base: Hit@5 = **0.71**, Hit@10 = 0.82,
MRR@10 = 0.51 (NLM summaries); about 4 ms per search.

**Abstention**: with γ = 0.40, 96% of out-of-scope questions are rejected while
96% of real consumer questions are still answered.

**End-to-end pilot** (Gemini 3.5 Flash-Lite, 48 requests):

| Route | Median latency |
|---|---|
| Cache hit | 5.9 ms |
| RAG answer | 4.1 s |
| LLM only (no evidence) | 3.3 s |

85% of RAG answers passed validation, and 98.5% of answer sentences were
supported by their cited passages.

Details and limitations are in [`docs/MIDSEM_PROGRESS.md`](docs/MIDSEM_PROGRESS.md).

## Tech stack

| Part | Choice | Why |
|---|---|---|
| LLM | Gemini 3.5 Flash-Lite (free tier) through an OpenAI-compatible API | lightweight and fast; switch to Groq or a local Ollama `qwen2.5:3b` by changing 3 env vars |
| Embeddings | `all-MiniLM-L6-v2`, ONNX quantised (23 MB) | runs on CPU with no PyTorch, small enough for serverless |
| Vector search | NumPy dot product (exact) | 27k vectors take about 1 ms; FAISS is only worth it at millions |
| Knowledge base | MedQuAD: 15,809 NIH answers in 9 subsets | trusted sources, CC BY 4.0 |
| API / UI | FastAPI + plain HTML/CSS/JS | no build step; deploys to Vercel as-is |
| Tests | pytest (58 tests) + GitHub Actions | fake encoder and fake LLM, so no API key is needed |

## Project layout

```
medrac_qa/            the framework (each file is one step of the pipeline)
  safety.py           emergency + never-cache rules
  query_analysis.py   intent, clinical qualifiers, topic words; compatible()
  embedder.py         MiniLM via ONNX Runtime (tokenise → encode → mean-pool → normalise)
  cache.py            CAG: semantic cache with the 4-check gate (+ exact/naive baselines),
                      memory / file / shared Redis storage
  knowledge.py        RAG: load index, cosine top-k
  evidence.py         evidence gate + sentence-level citation validation
  llm.py, prompts.py  OpenAI-compatible client and prompt templates
  pipeline.py         the MedRAC-QA algorithm (modes: full / rag / llm_only)
  service.py          builds a ready pipeline from settings
app.py                FastAPI app (Vercel entrypoint)
public/               web UI (index.html, style.css, app.js)
data/index/           prebuilt knowledge base (passages + embeddings + manifest)
models/minilm/        embedding model (ONNX) + tokenizer
scripts/              download_data.py, build_index.py, ask.py (CLI)
experiments/          eval_retrieval.py, eval_cache.py, eval_abstention.py, pilot_end_to_end.py
tests/                unit + API tests
docs/                 synopsis, mid-semester progress audit
```

## Run it locally

Requires Python 3.10 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (Linux/macOS: source .venv/bin/activate)
pip install -r requirements-dev.txt
copy .env.example .env            # then put your Gemini API key in .env
uvicorn app:app --reload          # open http://127.0.0.1:8000
```

Command line:

```bash
python scripts/ask.py "What are the symptoms of anemia?"
python scripts/ask.py --mode llm_only "What causes asthma?"
pytest -q
```

Rebuild the knowledge base and rerun the experiments (downloads about 30 MB of data):

```bash
python scripts/download_data.py
python scripts/build_index.py              # ~8 min on a laptop CPU
python experiments/eval_retrieval.py
python experiments/eval_cache.py
python experiments/eval_abstention.py
python experiments/pilot_end_to_end.py     # uses the LLM (about 36 calls)
```

## Deploy on Vercel

The repository is ready to deploy as is: `app.py` is detected as a FastAPI app,
and `public/` is served by the CDN.

1. Vercel → **Add New → Project** → import this GitHub repository (framework preset is detected automatically).
2. **Settings → Environment Variables**, add:
   - `LLM_API_KEY`: your Gemini API key
   - `LLM_MODEL` = `gemini-3.5-flash-lite`
   - `LLM_REASONING_EFFORT` = `minimal`
   - (`LLM_BASE_URL` defaults to Gemini's OpenAI-compatible endpoint)
3. Deploy. Check `https://<your-app>.vercel.app/api/health`.

Notes:
- The function bundle is about 235 MB (NumPy + ONNX Runtime + index + model), within Vercel's limit.
- `vercel.json` excludes tests, docs and raw data from the bundle.
- **Shared cache (recommended):** Vercel → your project → **Storage** → **Create
  Database** → **Upstash for Redis** (free plan) → connect it to the project, then
  redeploy. Vercel adds the Redis URL and token as environment variables and the
  app uses them automatically; `/api/health` then shows `"backend": "redis"`.
  Without it, each serverless instance keeps its own in-memory cache.
- Rate limit: 20 questions per minute per IP (`RATE_LIMIT_PER_MIN`).

## Configuration

All settings are environment variables (see [`.env.example`](.env.example)):

| Variable | Default | Meaning |
|---|---|---|
| `LLM_BASE_URL` / `LLM_API_KEY` / `LLM_MODEL` | Gemini 3.5 Flash-Lite | any OpenAI-compatible chat API |
| `TOP_K` | 5 | passages retrieved per question |
| `CACHE_THRESHOLD` (τ) | 0.85 | minimum question similarity for cache reuse (chosen in `eval_cache.py`) |
| `EVIDENCE_THRESHOLD` (γ) | 0.40 | minimum retrieval score before answering (chosen in `eval_abstention.py`) |
| `SUPPORT_THRESHOLD` (δ) | 0.40 | sentence-to-passage similarity counted as support |
| `CACHE_TTL_DAYS` | 30 | cached answers expire after this many days |
| `UPSTASH_REDIS_REST_URL` / `_TOKEN` (or `KV_REST_API_URL` / `_TOKEN`) | – | shared Redis cache for all server instances |
| `CACHE_FILE` | (memory) | path to persist the cache as JSON lines (local runs) |

## Roadmap (to 15 November 2026)

- [ ] Full comparison of baselines B1–B6 on a few hundred questions: answer quality (ROUGE-L, semantic similarity), citation precision, latency and cost, with paired bootstrap / McNemar tests
- [ ] PubMedQA (yes/no/maybe) accuracy and macro-F1: LLM-only vs evidence-grounded
- [ ] Hybrid retrieval (BM25 + dense), and a comparison with a biomedical embedding model
- [ ] NLI-based claim verification in place of the similarity check
- [x] Shared persistent cache (Upstash Redis) for serverless deployments
- [ ] Safety test set (emergency and never-cache recall)
- [ ] Knowledge-preloading CAG variant (context caching) as an experiment
- [ ] Final report and demo video

## Data and model credits

- **MedQuAD**: A. Ben Abacha and D. Demner-Fushman, "A Question-Entailment Approach to
  Question Answering", *BMC Bioinformatics* 20, 511 (2019). CC BY 4.0. The three subsets
  whose answers were removed for copyright reasons (A.D.A.M., MedlinePlus Drugs, Herbs)
  are not used.
- **TREC 2017 LiveQA-Med** test questions and judgements are used for evaluation only
  (downloaded by the script, not redistributed).
- **all-MiniLM-L6-v2**: sentence-transformers, Apache 2.0.
