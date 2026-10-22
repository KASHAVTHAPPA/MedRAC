"""RQ2/RQ3: does the cache gate stop unsafe reuse without losing useful hits?

Workload (built from MedQuAD, seeded, fully reproducible):
  * 300 "base" questions (one per disease/topic) are put in the cache.
  * For each base we ask four probe questions:
      paraphrase   same meaning, different words      -> SHOULD hit its base
      context      paraphrase + "in children" / "during pregnancy" / "in older adults"
                                                       -> must NOT reuse any answer
      intent       same topic, different question type (symptoms -> treatment)
      near_topic   same question type about the most similar *other* topic
                   (e.g. "Anemia" vs "Fanconi anemia")
  * A probe's ground truth is (topic, question type, clinical context). A hit is
    correct only if the reused answer has exactly that ground truth.

Compared caches (the same MiniLM embeddings for all):
  exact     normalised string match                    (baseline B3)
  naive     cosine similarity >= tau + freshness       (baseline B4)
  semantic  naive + context gate (proposed MedRAC-QA)  (B6's cache)

tau is chosen on the dev half of the bases (best F1 with unsafe-hit rate
<= 1%) and reported on the held-out test half.

    python experiments/eval_cache.py
"""

from __future__ import annotations

import random
import re
import time
from collections import defaultdict

import numpy as np
from common import load_kb_and_encoder, load_medquad_questions, save

from medrac_qa.cache import SemanticCache
from medrac_qa.query_analysis import analyze

SEED = 7
N_BASES = 300
TAUS = (0.75, 0.80, 0.85, 0.90, 0.95)
MAX_UNSAFE = 0.01

PARAPHRASES = {
    "information": ["Can you tell me what {x} is?", "Please explain {x}.", "Give me an overview of {x}."],
    "symptoms": ["What signs and symptoms does {x} cause?", "What are the common signs of {x}?", "Which symptoms are seen in {x}?"],
    "treatment": ["How is {x} treated?", "What treatment options are available for {x}?", "How can {x} be managed?"],
    "inheritance": ["Can {x} be passed down in families?", "Is {x} hereditary?", "Does {x} run in families?"],
    "frequency": ["How common is {x}?", "What is the prevalence of {x}?"],
    "causes": ["What is the cause of {x}?", "Why does {x} happen?", "What are the causes of {x}?"],
    "exams and tests": ["How is {x} diagnosed?", "What tests are used to diagnose {x}?"],
    "outlook": ["What is the prognosis of {x}?", "What is the long-term outlook for {x}?"],
    "prevention": ["How can {x} be prevented?", "What can be done to prevent {x}?"],
}
CONTEXTS = {"children": " in children", "pregnancy": " during pregnancy", "elderly": " in older adults"}


def same_topic(name: str) -> str:
    """Ground-truth key for a topic. MedQuAD sometimes lists one disease under
    two spellings ("Beta-thalassemia" / "beta thalassemia", "type IV" / "type 4");
    these are the same topic, so reusing an answer between them is correct."""
    s = name.lower()
    for roman, digit in (("iv", "4"), ("iii", "3"), ("ii", "2"), ("i", "1")):
        s = re.sub(rf"\btype {roman}\b", f"type {digit}", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s[:-1] if s.endswith("s") else s


def build_workload(rng: random.Random, encoder) -> tuple[list[dict], list[dict]]:
    rows = load_medquad_questions()
    canonical: dict[tuple[str, str], str] = {}
    for r in rows:
        if r["qtype"] in PARAPHRASES and len(r["focus"]) < 60:
            canonical.setdefault((r["focus"], r["qtype"]), r["question"])
    by_focus = defaultdict(list)
    for focus, qtype in canonical:
        by_focus[focus].append(qtype)

    # Foci with at least two question types, so an "intent" probe exists.
    foci = sorted(f for f, types in by_focus.items() if len(types) >= 2)
    rng.shuffle(foci)
    chosen = foci[:N_BASES]

    # Nearest *other* focus by embedding of the focus name, for near-topic probes.
    all_foci = sorted(by_focus)
    fvecs = encoder.encode(all_foci)
    index = {f: i for i, f in enumerate(all_foci)}

    bases, probes = [], []
    for focus in chosen:
        qtype = rng.choice(by_focus[focus])
        base_q = canonical[(focus, qtype)]
        bases.append({"question": base_q, "key": (same_topic(focus), qtype, "")})

        para = rng.choice(PARAPHRASES[qtype]).format(x=focus)
        probes.append({"kind": "paraphrase", "question": para, "key": (same_topic(focus), qtype, "")})

        ctx, suffix = rng.choice(list(CONTEXTS.items()))
        probes.append({"kind": "context", "question": para.rstrip("?.") + suffix + "?", "key": (same_topic(focus), qtype, ctx)})

        other = rng.choice([t for t in by_focus[focus] if t != qtype])
        probes.append({"kind": "intent", "question": canonical[(focus, other)], "key": (same_topic(focus), other, "")})

        sims = fvecs @ fvecs[index[focus]]
        for j in np.argsort(-sims):
            near = all_foci[j]
            if same_topic(near) != same_topic(focus) and (near, qtype) in canonical:
                probes.append({"kind": "near_topic", "question": canonical[(near, qtype)], "key": (same_topic(near), qtype, "")})
                break
    return bases, probes


def run(mode: str, tau: float, bases, probes, vec) -> dict:
    cache = SemanticCache("eval", threshold=tau, mode=mode)
    key_of = {}
    for b in bases:
        e = cache.store(b["question"], vec[b["question"]], analyze(b["question"]), "answer", [{"id": "x"}])
        key_of[e.id] = b["key"]
    cached_keys = set(key_of.values())

    per_kind = defaultdict(lambda: {"n": 0, "correct": 0, "wrong": 0})
    should_hit = correct = wrong = 0
    lookup_ms = []
    for p in probes:
        start = time.perf_counter()
        d = cache.lookup(p["question"], vec[p["question"]], analyze(p["question"]))
        lookup_ms.append((time.perf_counter() - start) * 1000)
        k = per_kind[p["kind"]]
        k["n"] += 1
        if p["key"] in cached_keys:
            should_hit += 1
        if d.hit:
            if key_of[d.entry.id] == p["key"]:
                correct += 1
                k["correct"] += 1
            else:
                wrong += 1
                k["wrong"] += 1

    hits = correct + wrong
    recall = correct / should_hit if should_hit else 0.0
    precision = correct / hits if hits else 1.0
    return {
        "mode": mode,
        "tau": tau,
        "useful_hit_rate": round(recall, 3),          # correct hits / probes that should hit
        "hit_precision": round(precision, 3),         # correct hits / all hits
        "unsafe_hit_rate": round(wrong / len(probes), 4),  # wrong reuse / all probes
        "f1": round(2 * precision * recall / (precision + recall), 3) if precision + recall else 0.0,
        "wrong_hits": wrong,
        "per_kind": {kind: dict(v) for kind, v in per_kind.items()},
        "median_lookup_ms": round(float(np.median(lookup_ms)), 3),
    }


def choose(results: list[dict]) -> dict:
    safe = [r for r in results if r["unsafe_hit_rate"] <= MAX_UNSAFE] or results
    return max(safe, key=lambda r: (r["f1"], -r["unsafe_hit_rate"]))


def main() -> None:
    rng = random.Random(SEED)
    _, encoder = load_kb_and_encoder()
    bases, probes = build_workload(rng, encoder)
    texts = sorted({b["question"] for b in bases} | {p["question"] for p in probes})
    vec = dict(zip(texts, encoder.encode(texts)))
    print(f"{len(bases)} cached questions, {len(probes)} probes")

    half = len(bases) // 2
    dev_keys = {b["key"][0] for b in bases[:half]}
    split = {
        "dev": ([b for b in bases if b["key"][0] in dev_keys], [p for i, p in enumerate(probes) if i // 4 < half]),
        "test": ([b for b in bases if b["key"][0] not in dev_keys], [p for i, p in enumerate(probes) if i // 4 >= half]),
    }

    report = {"seed": SEED, "bases": len(bases), "probes": len(probes), "max_unsafe": MAX_UNSAFE, "results": {}}
    for mode in ("exact", "naive", "semantic"):
        taus = (1.0,) if mode == "exact" else TAUS
        dev = [run(mode, t, *split["dev"], vec) for t in taus]
        best = choose(dev)
        test = run(mode, best["tau"], *split["test"], vec)
        report["results"][mode] = {"dev_sweep": dev, "chosen_tau": best["tau"], "test": test}
        print(f"{mode:9s} tau={best['tau']:.2f}  useful hits {test['useful_hit_rate']:.1%}  "
              f"precision {test['hit_precision']:.1%}  unsafe {test['unsafe_hit_rate']:.2%}  F1 {test['f1']:.3f}")

    report["examples"] = [{"kind": p["kind"], "question": p["question"]} for p in probes[:8]]
    print("saved", save("cache_gate", report))


if __name__ == "__main__":
    main()
