"""Build the RAG knowledge base from MedQuAD.

    python scripts/build_index.py

Steps:
  1. Parse the MedQuAD XML files (9 subsets that include answers).
  2. Drop empty and exactly duplicated answers.
  3. Split long answers into overlapping 180-word chunks.
  4. Embed "question + chunk" with all-MiniLM-L6-v2.
  5. Save passages, embeddings (float16) and a manifest to data/index/.

It also writes data/eval/medquad_questions.jsonl, the list of MedQuAD questions
used by the cache experiments.

Licence: MedQuAD is CC BY 4.0 (Ben Abacha & Demner-Fushman, 2019). The three
MedlinePlus subsets whose answers were removed for copyright reasons (A.D.A.M.,
Drugs, Herbs & Supplements) are skipped.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from medrac_qa.config import Settings  # noqa: E402
from medrac_qa.embedder import MiniLMEmbedder  # noqa: E402
from medrac_qa.text import chunk_words, clean  # noqa: E402

RAW = ROOT / "data" / "raw" / "MedQuAD"
OUT = ROOT / "data" / "index"
EVAL = ROOT / "data" / "eval"
SKIP = {"10_MPlus_ADAM_QA", "11_MPlusDrugs_QA", "12_MPlusHerbsSupplements_QA"}
CHUNK_WORDS, OVERLAP = 180, 30


def parse_medquad() -> list[dict]:
    """One record per QA pair that has an answer."""
    pairs = []
    for folder in sorted(p for p in RAW.iterdir() if p.is_dir() and p.name not in SKIP):
        for path in sorted(folder.glob("*.xml")):
            doc = ET.parse(path).getroot()
            source, doc_id = doc.get("source"), doc.get("id")
            focus = clean(doc.findtext("Focus") or "")
            for qa in doc.iter("QAPair"):
                q = qa.find("Question")
                answer = clean(qa.findtext("Answer") or "")
                if q is None or not answer:
                    continue
                pairs.append(
                    {
                        "answer_id": f"{source}_{doc_id}_Sec{qa.get('pid')}",
                        "qid": q.get("qid"),
                        "question": clean(q.text or ""),
                        "qtype": q.get("qtype") or "information",
                        "focus": focus,
                        "answer": answer,
                        "url": doc.get("url") or "",
                        "source": source,
                    }
                )
    return pairs


def main() -> None:
    t0 = time.time()
    pairs = parse_medquad()
    print(f"QA pairs with answers: {len(pairs)}")

    # Exact duplicates (the same answer text published by two NIH sites) are
    # kept once; the alias map lets the evaluation still recognise both ids.
    seen: dict[str, str] = {}
    aliases: dict[str, str] = {}
    unique = []
    for p in pairs:
        key = hashlib.sha1(p["answer"].lower().encode()).hexdigest()
        if key in seen:
            aliases[p["answer_id"]] = seen[key]
            continue
        seen[key] = p["answer_id"]
        unique.append(p)
    print(f"unique answers: {len(unique)} ({len(aliases)} duplicates removed)")

    passages, embed_texts = [], []
    for p in unique:
        for i, chunk in enumerate(chunk_words(p["answer"], CHUNK_WORDS, OVERLAP)):
            passages.append(
                {
                    "id": f"{p['answer_id']}_c{i}",
                    "answer_id": p["answer_id"],
                    "focus": p["focus"],
                    "qtype": p["qtype"],
                    "question": p["question"],
                    "text": chunk,
                    "url": p["url"],
                    "source": p["source"],
                }
            )
            # Prefixing the question ("document expansion") tells the encoder
            # what the chunk is about even when the chunk never names the topic.
            embed_texts.append(f"{p['question']} {chunk}")
    print(f"passages: {len(passages)}")

    settings = Settings()
    embedder = MiniLMEmbedder(settings.model_dir)
    vectors = []
    for start in range(0, len(embed_texts), 512):
        vectors.append(embedder.encode(embed_texts[start : start + 512], batch_size=64))
        print(f"  embedded {min(start + 512, len(embed_texts))}/{len(embed_texts)}", end="\r")
    matrix = np.vstack(vectors).astype(np.float16)
    print()

    version = hashlib.sha1("".join(p["id"] + p["text"] for p in passages).encode()).hexdigest()[:12]
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(OUT / "passages.json.gz", "wt", encoding="utf-8") as f:
        json.dump(passages, f, ensure_ascii=False)
    np.save(OUT / "embeddings.npy", matrix)
    (OUT / "manifest.json").write_text(
        json.dumps(
            {
                "version": version,
                "dataset": "MedQuAD (CC BY 4.0), 9 subsets with answers",
                "qa_pairs": len(pairs),
                "unique_answers": len(unique),
                "passages": len(passages),
                "chunk_words": CHUNK_WORDS,
                "chunk_overlap": OVERLAP,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2 (ONNX, quint8)",
                "embedding_text": "question + ' ' + chunk",
                "built": time.strftime("%Y-%m-%d"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (OUT / "answer_aliases.json").write_text(json.dumps(aliases), encoding="utf-8")

    EVAL.mkdir(parents=True, exist_ok=True)
    with (EVAL / "medquad_questions.jsonl").open("w", encoding="utf-8") as f:
        for p in pairs:
            row = {k: p[k] for k in ("qid", "question", "qtype", "focus", "answer_id", "source")}
            row["answer_id"] = aliases.get(row["answer_id"], row["answer_id"])
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    size = sum(f.stat().st_size for f in OUT.iterdir()) / 1e6
    print(f"index version {version}, {size:.1f} MB, {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
