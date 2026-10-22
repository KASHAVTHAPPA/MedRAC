"""Evidence checks before and after generation.

Before:  sufficiency gate. If even the best passage is only weakly similar to
         the question (score < gamma), the knowledge base probably does not
         cover it, so we abstain instead of letting the LLM guess.

After:   citation validation. The answer is split into sentences. A sentence
         counts as supported by the passages it cites when EITHER
           - semantic: its embedding is at least delta-similar to one of them, OR
           - lexical:  at least 80% of its content words appear in one of them.
         The lexical test matters for short list items ("Dizziness [1]"),
         whose embeddings are not comparable with a whole paragraph.
         The answer is accepted only if it cites something, cites only
         passages that exist, and at most `max_unsupported_rate` of its
         sentences are unsupported.

Both tests approximate support; neither proves entailment. An NLI model is
the planned upgrade (see docs/MIDSEM_PROGRESS.md).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from .embedder import TextEncoder
from .knowledge import Hit
from .prompts import INSUFFICIENT
from .query_analysis import STOPWORDS
from .text import citations_in, split_sentences, strip_citations

LEXICAL_MIN = 0.8
_WORD = re.compile(r"[a-z0-9]+")


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in STOPWORDS}


def lexical_support(sentence: str, passage_text: str) -> float:
    """Share of the sentence's content words that occur in the passage."""
    words = _content_words(sentence)
    if not words:
        return 0.0
    passage = passage_text.lower()
    return sum(w in passage or w.rstrip("s") in passage for w in words) / len(words)


def sufficiency(hits: list[Hit], gamma: float) -> tuple[bool, float]:
    best = hits[0].score if hits else 0.0
    return best >= gamma, best


@dataclass
class SentenceCheck:
    sentence: str
    cited: list[int]
    support: float      # best embedding similarity to a cited passage
    lexical: float      # best content-word overlap with a cited passage
    supported: bool


@dataclass
class Validation:
    passed: bool
    reason: str
    sentences: list[SentenceCheck] = field(default_factory=list)
    unsupported_rate: float = 0.0
    invalid_citations: list[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reason": self.reason,
            "unsupported_rate": round(self.unsupported_rate, 3),
            "invalid_citations": self.invalid_citations,
            "sentences": [
                {
                    "text": s.sentence,
                    "cited": s.cited,
                    "support": round(s.support, 3),
                    "lexical": round(s.lexical, 3),
                    "supported": s.supported,
                }
                for s in self.sentences
            ],
        }


def validate_answer(
    answer: str,
    passage_texts: list[str],
    passage_vecs: np.ndarray,
    encoder: TextEncoder,
    delta: float,
    max_unsupported_rate: float,
) -> Validation:
    """passage_texts[i] / passage_vecs[i] belong to evidence passage [i+1]."""
    if INSUFFICIENT in answer:
        return Validation(False, "model reported insufficient evidence")

    # Lead-in lines such as "Other symptoms include:" make no claim of their own.
    sentences = [
        s for s in split_sentences(answer)
        if len(strip_citations(s)) > 3 and not strip_citations(s).endswith(":")
    ]
    if not sentences:
        return Validation(False, "empty answer")

    n = len(passage_vecs)
    all_cited = [c for s in sentences for c in citations_in(s)]
    invalid = sorted({c for c in all_cited if c < 1 or c > n})
    if not all_cited:
        return Validation(False, "answer has no citations")

    vecs = encoder.encode([strip_citations(s) for s in sentences])
    checks = []
    for sentence, vec in zip(sentences, vecs):
        cited = [c for c in citations_in(sentence) if 1 <= c <= n]
        # A sentence without its own citation may lean on the passages cited
        # elsewhere in the answer (common in short, two-sentence answers).
        pool = cited or sorted({c for c in all_cited if 1 <= c <= n})
        support = float(max(passage_vecs[c - 1] @ vec for c in pool)) if pool else 0.0
        lexical = max((lexical_support(strip_citations(sentence), passage_texts[c - 1]) for c in pool), default=0.0)
        checks.append(SentenceCheck(sentence, cited, support, lexical, support >= delta or lexical >= LEXICAL_MIN))

    rate = sum(not c.supported for c in checks) / len(checks)
    if invalid:
        return Validation(False, f"cites passages that do not exist: {invalid}", checks, rate, invalid)
    if rate > max_unsupported_rate:
        return Validation(False, f"{rate:.0%} of sentences are not supported by the evidence", checks, rate)
    return Validation(True, "all checks passed", checks, rate)
