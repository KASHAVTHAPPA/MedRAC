"""Shared helpers for the experiment scripts (run from the project root)."""

from __future__ import annotations

import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from medrac_qa.config import Settings  # noqa: E402
from medrac_qa.embedder import MiniLMEmbedder  # noqa: E402
from medrac_qa.knowledge import KnowledgeBase  # noqa: E402

RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "experiments" / "results"
LIVEQA_XML = RAW / "liveqa" / "TREC-2017-LiveQA-Medical-Test-Questions-w-summaries.xml"
LIVEQA_QRELS = next((RAW / "liveqa").glob("**/All-qrels_*.txt"), None) if (RAW / "liveqa").exists() else None


def load_kb_and_encoder() -> tuple[KnowledgeBase, MiniLMEmbedder]:
    s = Settings()
    return KnowledgeBase.load(s.index_dir), MiniLMEmbedder(s.model_dir)


def load_medquad_questions() -> list[dict]:
    path = ROOT / "data" / "eval" / "medquad_questions.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_liveqa() -> list[dict]:
    """TREC 2017 LiveQA-Med test questions: real questions sent to the NLM by the public."""
    rows = []
    for q in ET.parse(LIVEQA_XML).getroot().iter("NLM-QUESTION"):
        orig = q.find("Original-Question")
        subject = (orig.findtext("SUBJECT") or "").strip()
        message = (orig.findtext("MESSAGE") or "").strip()
        rows.append(
            {
                "qid": q.get("qid"),                      # "TQ1"
                "original": f"{subject}. {message}".strip(". "),
                "summary": (q.findtext("NLM-Summary") or "").strip(),
            }
        )
    return rows


def load_liveqa_qrels(min_grade: int = 3) -> dict[str, set[str]]:
    """Judged MedQuAD answers per LiveQA question (grade 3 = incomplete, 4 = excellent)."""
    rel: dict[str, set[str]] = {}
    for line in LIVEQA_QRELS.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 3:
            continue
        qnum, grade, answer = parts
        if int(grade.split("-")[0]) >= min_grade:
            rel.setdefault(f"TQ{qnum}", set()).add(answer.removesuffix(".txt"))
    return rel


def save(name: str, data: dict) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path
