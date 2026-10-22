"""RQ1 (retrieval part): does dense retrieval find the right NIH answer for real consumer questions?

Test set: TREC 2017 LiveQA-Med (104 real questions sent to the U.S. National
Library of Medicine). Ben Abacha & Demner-Fushman (2019) had experts judge
2,479 MedQuAD answers for these questions. An answer judged 3 (incomplete)
or 4 (excellent) counts as relevant.

Only questions with at least one relevant answer inside our knowledge base
are scored (most judged answers come from the three MedlinePlus subsets whose
text is copyrighted and therefore excluded).

Metrics: Hit@k (at least one relevant answer in the top k), Recall@k and
MRR@10. Both the raw consumer message and the NLM's one-line summary of it
are evaluated.

    python experiments/eval_retrieval.py
"""

from __future__ import annotations

import json
import time

import numpy as np
from common import ROOT, load_kb_and_encoder, load_liveqa, load_liveqa_qrels, save

KS = (1, 3, 5, 10)


def ranked_answers(kb, vec, depth=50) -> list[str]:
    """Distinct answer ids in rank order (several chunks may share one answer)."""
    seen: list[str] = []
    for hit in kb.search(vec, depth):
        if hit.passage.answer_id not in seen:
            seen.append(hit.passage.answer_id)
    return seen[:10]


def evaluate(kb, encoder, questions, field, relevant) -> dict:
    hits = {k: [] for k in KS}
    recall = {k: [] for k in KS}
    rr, latency = [], []
    for q in questions:
        rel = relevant[q["qid"]]
        start = time.perf_counter()
        ranked = ranked_answers(kb, encoder.encode([q[field]])[0])
        latency.append((time.perf_counter() - start) * 1000)
        for k in KS:
            found = rel & set(ranked[:k])
            hits[k].append(bool(found))
            recall[k].append(len(found) / len(rel))
        rank = next((i for i, a in enumerate(ranked, 1) if a in rel), None)
        rr.append(1 / rank if rank else 0.0)
    return {
        **{f"hit@{k}": round(float(np.mean(hits[k])), 3) for k in KS},
        **{f"recall@{k}": round(float(np.mean(recall[k])), 3) for k in KS},
        "mrr@10": round(float(np.mean(rr)), 3),
        "median_latency_ms": round(float(np.median(latency)), 1),
    }


def main() -> None:
    kb, encoder = load_kb_and_encoder()
    in_kb = {p.answer_id for p in kb.passages}
    aliases = json.loads((ROOT / "data" / "index" / "answer_aliases.json").read_text())

    relevant = {}
    for qid, answers in load_liveqa_qrels().items():
        reachable = {aliases.get(a, a) for a in answers} & in_kb
        if reachable:
            relevant[qid] = reachable
    questions = [q for q in load_liveqa() if q["qid"] in relevant]
    print(f"LiveQA questions with a relevant answer in the KB: {len(questions)} of 104")

    result = {
        "dataset": "TREC 2017 LiveQA-Med test questions, MedQuAD judged answers (grade >= 3)",
        "questions_evaluated": len(questions),
        "knowledge_base": {"passages": len(kb), "version": kb.version},
        "original_question": evaluate(kb, encoder, questions, "original", relevant),
        "nlm_summary": evaluate(kb, encoder, questions, "summary", relevant),
    }
    print(json.dumps(result, indent=2))
    print("saved", save("retrieval_liveqa", result))


if __name__ == "__main__":
    main()
