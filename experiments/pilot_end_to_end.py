"""Pilot end-to-end run with the real LLM (small, to respect free-tier limits).

For 12 common health questions, in this order:
  1. the question                      -> expected route: rag   (cache is empty)
  2. a paraphrase of it                -> expected route: cache (validated reuse)
  3. the paraphrase "in children" etc. -> expected route: rag   (gate rejects reuse)
plus the same 12 questions in LLM-only mode (baseline B1, no evidence).

Reports route counts, median latency per route, validation pass rate and the
share of answer sentences supported by evidence. This is a PILOT for the
mid-semester demo; the full comparison (B1-B6 on a few hundred questions,
answer-quality metrics and significance tests) is phase 2.

    python experiments/pilot_end_to_end.py
"""

from __future__ import annotations

import statistics
import time
from collections import Counter, defaultdict

from common import save

from medrac_qa.service import build_pipeline

QUESTIONS = [
    ("What are the symptoms of anemia?", "What signs and symptoms does anemia cause?", " in children"),
    ("How is glaucoma diagnosed?", "What tests are used to diagnose glaucoma?", " in older adults"),
    ("What causes high blood pressure?", "What are the causes of high blood pressure?", " during pregnancy"),
    ("What are the treatments for asthma?", "How is asthma treated?", " in children"),
    ("What is psoriasis?", "Can you tell me what psoriasis is?", " in children"),
    ("How to prevent osteoporosis?", "How can osteoporosis be prevented?", " in older adults"),
    ("What are the symptoms of hypothyroidism?", "What are the common signs of hypothyroidism?", " during pregnancy"),
    ("What is the outlook for kidney failure?", "What is the prognosis of kidney failure?", " in older adults"),
    ("What are the treatments for depression?", "How is depression treated?", " during pregnancy"),
    ("What causes gout?", "What is the cause of gout?", " in older adults"),
    ("What are the symptoms of celiac disease?", "Which symptoms are seen in celiac disease?", " in children"),
    ("What is glaucoma?", "Please explain glaucoma.", " in children"),
]
PAUSE_S = 4.0  # stay under the free-tier requests-per-minute limit


def main() -> None:
    pipeline = build_pipeline()
    pipeline.cache.clear()
    rows = []

    def run(question: str, mode: str, kind: str) -> None:
        r = pipeline.answer(question, mode)
        v = r.validation or {}
        sentences = v.get("sentences", [])
        rows.append(
            {
                "kind": kind,
                "mode": mode,
                "question": question,
                "route": r.route,
                "total_ms": r.timings_ms.get("total"),
                "validation_passed": v.get("passed"),
                "supported_share": (sum(s["supported"] for s in sentences) / len(sentences)) if sentences else None,
                "cache_reason": (r.cache or {}).get("reason"),
            }
        )
        print(f"{kind:10s} {r.route:9s} {r.timings_ms.get('total'):>8} ms  {question}")
        if r.route not in ("cache", "emergency", "abstain"):
            time.sleep(PAUSE_S)

    for base, para, ctx in QUESTIONS:
        run(base, "full", "base")
        run(para, "full", "paraphrase")
        run(para.rstrip("?.") + ctx + "?", "full", "context")
    for base, _, _ in QUESTIONS:
        run(base, "llm_only", "llm_only")

    by_route = defaultdict(list)
    for r in rows:
        by_route[r["route"]].append(r["total_ms"])
    expected = {"base": "rag", "paraphrase": "cache", "context": "rag", "llm_only": "llm_only"}
    rag_rows = [r for r in rows if r["route"] in ("rag", "fallback")]
    summary = {
        "llm_model": pipeline.s.llm_model,
        "routes": dict(Counter(r["route"] for r in rows)),
        "as_expected": {k: sum(r["route"] == v for r in rows if r["kind"] == k) for k, v in expected.items()},
        "median_ms_by_route": {k: round(statistics.median(v), 1) for k, v in by_route.items()},
        "validation_pass_rate": round(sum(bool(r["validation_passed"]) for r in rag_rows) / len(rag_rows), 3) if rag_rows else None,
        "mean_supported_sentence_share": round(
            statistics.mean(r["supported_share"] for r in rag_rows if r["supported_share"] is not None), 3
        ) if rag_rows else None,
    }
    print(summary)
    print("saved", save("pilot_end_to_end", {"summary": summary, "rows": rows}))


if __name__ == "__main__":
    main()
