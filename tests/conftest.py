"""Test doubles: a deterministic fake encoder, a tiny knowledge base and a fake LLM.

They let every pipeline branch be tested in milliseconds, with no model
download and no API key.
"""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from medrac_qa.cache import SemanticCache  # noqa: E402
from medrac_qa.config import Settings  # noqa: E402
from medrac_qa.knowledge import KnowledgeBase, Passage  # noqa: E402
from medrac_qa.llm import LLMReply  # noqa: E402
from medrac_qa.pipeline import MedRACPipeline  # noqa: E402

DIM = 256
_WORD = re.compile(r"[a-z]+")
_SKIP = {"the", "what", "are", "who", "how", "does", "and", "for", "with"}


class FakeEncoder:
    """Bag-of-words hashed into 256 dimensions: texts sharing words get similar vectors."""

    dim = DIM

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for i, t in enumerate(texts):
            for w in _WORD.findall(t.lower()):
                if len(w) > 2 and w not in _SKIP:
                    out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.clip(norms, 1e-9, None)


class FakeLLM:
    model = "fake-llm"

    def __init__(self, reply: str = "Anemia causes fatigue and weakness [1].") -> None:
        self.reply = reply
        self.calls = 0

    def chat(self, messages, temperature=0.1, retries=2):
        self.calls += 1
        return LLMReply(self.reply, self.model, 1.0)


PASSAGES = [
    ("anemia", "symptoms", "What are the symptoms of anemia?", "Anemia causes fatigue weakness and pale skin."),
    ("glaucoma", "diagnosis", "How is glaucoma diagnosed?", "Glaucoma is diagnosed with a dilated eye exam and tonometry."),
    ("asthma", "treatment", "What are the treatments for asthma?", "Asthma is treated with inhaled corticosteroids and quick relief inhalers."),
]


@pytest.fixture
def encoder() -> FakeEncoder:
    return FakeEncoder()


@pytest.fixture
def kb(encoder) -> KnowledgeBase:
    passages = [
        Passage(f"P{i}_c0", f"P{i}", focus, qtype, q, text, f"https://example.org/{focus}", "TEST")
        for i, (focus, qtype, q, text) in enumerate(PASSAGES)
    ]
    vecs = encoder.encode([f"{p.question} {p.text}" for p in passages])
    return KnowledgeBase(passages, vecs, version="v1")


@pytest.fixture
def settings() -> Settings:
    s = Settings()
    s.llm_api_key = "test-key"
    s.top_k = 2
    s.cache_threshold = 0.85
    s.evidence_threshold = 0.3
    s.support_threshold = 0.3
    return s


@pytest.fixture
def make_pipeline(settings, kb, encoder):
    def _make(reply: str = "Anemia causes fatigue and weakness [1].", cache_mode: str = "semantic"):
        cache = SemanticCache(kb.version, settings.cache_threshold, mode=cache_mode,
                              passage_exists=lambda pid: kb.get(pid) is not None)
        llm = FakeLLM(reply)
        return MedRACPipeline(settings, kb, encoder, cache, llm), llm

    return _make
