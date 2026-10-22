"""RAG retrieval: cosine-similarity search over the MedQuAD knowledge base.

The index is built offline by scripts/build_index.py and stored as:

    data/index/passages.json.gz   one record per passage (text + source metadata)
    data/index/embeddings.npy     one 384-d unit vector per passage (float16)
    data/index/manifest.json      corpus version, counts, build settings

With about 25k passages, exact search is one matrix-vector product
(25k x 384 multiply-adds, around a millisecond), so an approximate index such
as FAISS is unnecessary at this scale. It becomes worthwhile above about a
million passages.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Passage:
    id: str          # e.g. "GARD_0000004_Sec2_c0" = answer 2 of document 0000004, chunk 0
    answer_id: str   # MedQuAD QA-pair id, e.g. "GARD_0000004_Sec2" (matches LiveQA qrels)
    focus: str       # the disease / drug / topic the answer is about
    qtype: str       # MedQuAD question type, e.g. "symptoms"
    question: str    # the original MedQuAD question this answer belongs to
    text: str        # the answer chunk shown to the LLM
    url: str         # NIH page the answer comes from
    source: str      # e.g. "GARD", "CancerGov"

    def to_dict(self) -> dict:
        return self.__dict__.copy()


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float     # cosine similarity in [-1, 1]
    index: int = -1  # row in the embedding matrix


class KnowledgeBase:
    def __init__(self, passages: list[Passage], embeddings: np.ndarray, version: str) -> None:
        if len(passages) != embeddings.shape[0]:
            raise ValueError("passages and embeddings differ in length")
        self.passages = passages
        self.matrix = embeddings.astype(np.float32)
        self.version = version
        self._by_id = {p.id: p for p in passages}

    @classmethod
    def load(cls, index_dir: Path) -> "KnowledgeBase":
        manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
        with gzip.open(index_dir / "passages.json.gz", "rt", encoding="utf-8") as f:
            passages = [Passage(**row) for row in json.load(f)]
        embeddings = np.load(index_dir / "embeddings.npy")
        return cls(passages, embeddings, manifest["version"])

    def __len__(self) -> int:
        return len(self.passages)

    def get(self, passage_id: str) -> Passage | None:
        return self._by_id.get(passage_id)

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[Hit]:
        """Top-k passages by cosine similarity (vectors are unit length)."""
        scores = self.matrix @ query_vec.astype(np.float32)
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return [Hit(self.passages[i], float(scores[i]), int(i)) for i in top]

    def vectors(self, hits: list[Hit]) -> np.ndarray:
        return self.matrix[[h.index for h in hits]]
