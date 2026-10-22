"""Small text helpers shared by the index builder and the pipeline."""

from __future__ import annotations

import re

_SPACES = re.compile(r"\s+")
# Split after . ! ? when followed by whitespace and a capital letter or digit,
# so abbreviations such as "e.g. the" are not split.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_CITATION = re.compile(r"\[(\d+)\]")


def clean(text: str) -> str:
    """Collapse whitespace and strip."""
    return _SPACES.sub(" ", text or "").strip()


def normalize_question(question: str) -> str:
    """Canonical form used by the exact-match cache baseline."""
    q = clean(question).lower()
    q = re.sub(r"[^a-z0-9 ]", " ", q)
    return _SPACES.sub(" ", q).strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences; bullet lines count as sentences too."""
    parts: list[str] = []
    for line in re.split(r"\n+|\s-\s", text or ""):
        line = clean(re.sub(r"^\s*[-*•]\s*", "", line))
        if line:
            parts.extend(s for s in _SENTENCE_END.split(line) if s.strip())
    return parts


def chunk_words(text: str, size: int = 180, overlap: int = 30) -> list[str]:
    """Split text into overlapping windows of `size` words.

    180 words is roughly 230 word-piece tokens, which fits inside the 256-token
    window of all-MiniLM-L6-v2, so no part of a chunk is silently truncated
    when it is embedded.
    """
    if size <= overlap:
        raise ValueError("size must be larger than overlap")
    words = clean(text).split(" ")
    if len(words) <= size:
        return [" ".join(words)] if words != [""] else []
    chunks = []
    step = size - overlap
    for start in range(0, len(words), step):
        window = words[start : start + size]
        chunks.append(" ".join(window))
        if start + size >= len(words):
            break
    return chunks


def citations_in(text: str) -> list[int]:
    """Citation numbers such as [1] and [3] found in a sentence, in order."""
    return [int(n) for n in _CITATION.findall(text)]


def strip_citations(text: str) -> str:
    return clean(_CITATION.sub("", text))
