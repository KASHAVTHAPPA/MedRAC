"""CAG: a safety-constrained semantic answer cache.

Each entry is c = (e, a, E, t, v): question embedding e, validated answer a,
its evidence E, creation time t and knowledge-base version v.

A new question q may reuse entry c* (the most similar cached question) only
if ALL four checks pass. This "cache gate" is the main idea of the project:

    1. similarity   cos(q, e*) >= tau
    2. freshness    v* == current KB version and age(t*) <= TTL
    3. context      same intent, clinical qualifiers, numbers and topic words
                    (see query_analysis.compatible)
    4. evidence     every cited passage still exists in the knowledge base

Similarity alone is not enough: "symptoms of diabetes in children" and
"symptoms of diabetes in pregnancy" score above 0.9 but need different
answers. Check 3 catches this.

Modes (for the ablation experiments):
    "semantic"  all four checks          (proposed method)
    "naive"     similarity + freshness   (plain semantic cache baseline)
    "exact"     normalised string match  (exact-match cache baseline)
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .query_analysis import QueryProfile, compatible
from .text import normalize_question

DAY = 86_400.0


@dataclass
class CacheEntry:
    question: str
    embedding: list[float]
    answer: str
    citations: list[dict]          # passages shown with the answer
    profile: dict                  # QueryProfile.to_dict()
    kb_version: str
    created_at: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    hits: int = 0


@dataclass
class CacheDecision:
    hit: bool
    entry: CacheEntry | None = None
    similarity: float = 0.0
    reason: str = "cache empty"
    checks: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "hit": self.hit,
            "similarity": round(self.similarity, 4),
            "reason": self.reason,
            "checks": self.checks,
            "matched_question": self.entry.question if self.entry else None,
        }


class MemoryBackend:
    """Entries live in this process only (lost when a serverless instance stops)."""

    def load(self) -> list[CacheEntry]:
        return []

    def save(self, entry: CacheEntry) -> None:
        pass

    def clear(self) -> None:
        pass


class JsonFileBackend:
    """Entries are appended to a JSON-lines file, for local runs and experiments."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def load(self) -> list[CacheEntry]:
        if not self.path.exists():
            return []
        with self.path.open(encoding="utf-8") as f:
            return [CacheEntry(**json.loads(line)) for line in f if line.strip()]

    def save(self, entry: CacheEntry) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def clear(self) -> None:
        self.path.unlink(missing_ok=True)


class SemanticCache:
    def __init__(
        self,
        kb_version: str,
        threshold: float = 0.90,
        ttl_days: float = 30.0,
        mode: str = "semantic",
        backend: MemoryBackend | JsonFileBackend | None = None,
        passage_exists: Callable[[str], bool] = lambda _id: True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if mode not in ("semantic", "naive", "exact"):
            raise ValueError(f"unknown cache mode {mode}")
        self.kb_version = kb_version
        self.threshold = threshold
        self.ttl = ttl_days * DAY
        self.mode = mode
        self.backend = backend or MemoryBackend()
        self.passage_exists = passage_exists
        self.clock = clock
        self._lock = threading.Lock()
        self.entries: list[CacheEntry] = self.backend.load()
        self._matrix = self._build_matrix()
        self.stats = {"lookups": 0, "hits": 0, "stored": 0}

    def _build_matrix(self) -> np.ndarray:
        if not self.entries:
            return np.zeros((0, 384), dtype=np.float32)
        return np.array([e.embedding for e in self.entries], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.entries)

    # ------------------------------------------------------------------ lookup
    def lookup(self, question: str, q_vec: np.ndarray, profile: QueryProfile) -> CacheDecision:
        with self._lock:
            self.stats["lookups"] += 1
            if not self.entries:
                return CacheDecision(False)

            if self.mode == "exact":
                key = normalize_question(question)
                for e in self.entries:
                    if normalize_question(e.question) == key:
                        return self._accept(e, 1.0, {"exact_match": True})
                return CacheDecision(False, reason="no exact match")

            sims = self._matrix @ q_vec.astype(np.float32)
            best = int(np.argmax(sims))
            entry, sim = self.entries[best], float(sims[best])
            checks: dict = {"similarity": sim >= self.threshold}

            # 1. similarity
            if sim < self.threshold:
                return CacheDecision(False, entry, sim, f"similarity {sim:.2f} below {self.threshold:.2f}", checks)

            # 2. freshness
            fresh = entry.kb_version == self.kb_version and self.clock() - entry.created_at <= self.ttl
            checks["fresh"] = fresh
            if not fresh:
                return CacheDecision(False, entry, sim, "cached answer is out of date", checks)

            if self.mode == "semantic":
                # 3. clinical context
                ok, why = compatible(profile, QueryProfile.from_dict(entry.profile))
                checks["context"] = ok
                if not ok:
                    return CacheDecision(False, entry, sim, f"rejected: {why}", checks)
                # 4. evidence still present
                evidence_ok = bool(entry.citations) and all(self.passage_exists(c["id"]) for c in entry.citations)
                checks["evidence"] = evidence_ok
                if not evidence_ok:
                    return CacheDecision(False, entry, sim, "rejected: evidence no longer available", checks)

            return self._accept(entry, sim, checks)

    def _accept(self, entry: CacheEntry, sim: float, checks: dict) -> CacheDecision:
        entry.hits += 1
        self.stats["hits"] += 1
        return CacheDecision(True, entry, sim, "validated cache hit", checks)

    # ------------------------------------------------------------------- store
    def store(self, question: str, q_vec: np.ndarray, profile: QueryProfile, answer: str, citations: list[dict]) -> CacheEntry:
        entry = CacheEntry(
            question=question,
            embedding=[round(float(x), 5) for x in q_vec],
            answer=answer,
            citations=citations,
            profile=profile.to_dict(),
            kb_version=self.kb_version,
            created_at=self.clock(),
        )
        with self._lock:
            self.entries.append(entry)
            row = np.asarray(entry.embedding, dtype=np.float32)[None, :]
            self._matrix = row if len(self._matrix) == 0 else np.vstack([self._matrix, row])
            self.stats["stored"] += 1
        self.backend.save(entry)
        return entry

    def clear(self) -> None:
        with self._lock:
            self.entries = []
            self._matrix = self._build_matrix()
        self.backend.clear()

    def summary(self) -> dict:
        lookups = self.stats["lookups"]
        return {
            "mode": self.mode,
            "entries": len(self.entries),
            "threshold": self.threshold,
            **self.stats,
            "hit_rate": round(self.stats["hits"] / lookups, 3) if lookups else 0.0,
        }
