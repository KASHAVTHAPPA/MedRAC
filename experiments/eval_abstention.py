"""RQ4 (gate part): choosing gamma, the evidence threshold for abstaining.

If the best retrieved passage scores below gamma, the system abstains instead
of letting the LLM answer from weak evidence. A good gamma rejects
out-of-scope questions while still answering in-scope ones.

In-scope:      104 real consumer questions (TREC LiveQA-Med, NLM summaries)
               + 200 MedQuAD questions (seeded sample)
Out-of-scope:  50 non-medical questions (experiments/data/out_of_scope.txt)

Selection rule: the largest coverage of in-scope questions among the gammas
that reject at least 95% of out-of-scope questions.

    python experiments/eval_abstention.py
"""

from __future__ import annotations

import random

import numpy as np
from common import ROOT, load_kb_and_encoder, load_liveqa, load_medquad_questions, save

GAMMAS = [round(g, 2) for g in np.arange(0.30, 0.71, 0.05)]
MIN_REJECT = 0.95


def best_scores(kb, encoder, questions: list[str]) -> np.ndarray:
    vecs = encoder.encode(questions)
    return np.array([kb.search(v, 1)[0].score for v in vecs])


def main() -> None:
    kb, encoder = load_kb_and_encoder()
    rng = random.Random(11)
    liveqa = [q["summary"] for q in load_liveqa() if q["summary"]]
    medquad = [r["question"] for r in rng.sample(load_medquad_questions(), 200)]
    oos = [
        line.strip()
        for line in (ROOT / "experiments" / "data" / "out_of_scope.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]

    s_live, s_med, s_oos = (best_scores(kb, encoder, qs) for qs in (liveqa, medquad, oos))
    sweep = []
    for g in GAMMAS:
        sweep.append(
            {
                "gamma": g,
                "coverage_liveqa": round(float((s_live >= g).mean()), 3),
                "coverage_medquad": round(float((s_med >= g).mean()), 3),
                "reject_out_of_scope": round(float((s_oos < g).mean()), 3),
            }
        )
    ok = [r for r in sweep if r["reject_out_of_scope"] >= MIN_REJECT]
    chosen = max(ok, key=lambda r: (r["coverage_liveqa"] + r["coverage_medquad"], -r["gamma"])) if ok else None

    for r in sweep:
        mark = "  <- chosen" if chosen and r["gamma"] == chosen["gamma"] else ""
        print(f"gamma {r['gamma']:.2f}  LiveQA answered {r['coverage_liveqa']:.1%}  "
              f"MedQuAD answered {r['coverage_medquad']:.1%}  out-of-scope rejected {r['reject_out_of_scope']:.1%}{mark}")

    report = {
        "in_scope": {"liveqa": len(liveqa), "medquad": len(medquad)},
        "out_of_scope": len(oos),
        "score_summary": {
            name: {"median": round(float(np.median(s)), 3), "min": round(float(s.min()), 3), "max": round(float(s.max()), 3)}
            for name, s in (("liveqa", s_live), ("medquad", s_med), ("out_of_scope", s_oos))
        },
        "sweep": sweep,
        "rule": f"max coverage with out-of-scope rejection >= {MIN_REJECT:.0%}",
        "chosen_gamma": chosen["gamma"] if chosen else None,
        "highest_scoring_out_of_scope": sorted(zip(s_oos.round(3).tolist(), oos), reverse=True)[:5],
    }
    print("saved", save("abstention", report))


if __name__ == "__main__":
    main()
