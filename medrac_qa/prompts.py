"""Prompt templates. Kept in one file so they can be versioned and compared."""

from __future__ import annotations

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"

GROUNDED_SYSTEM = f"""You are a careful medical information assistant for the general public.
Answer the question using ONLY the numbered evidence passages provided.

Rules:
- Every sentence must end with the citation number(s) of the passage(s) that support it, like [1] or [2][3].
- Do not add facts that are not in the passages, even if you know them.
- If the passages do not contain the answer, reply with exactly: {INSUFFICIENT}
- Do not diagnose the user or prescribe medicines or doses.
- Use plain, simple language. Keep the answer under 150 words.
"""

LLM_ONLY_SYSTEM = """You are a careful medical information assistant for the general public.
Answer the question in plain, simple language in under 150 words.
Do not diagnose the user or prescribe medicines or doses."""


def grounded_messages(question: str, passages: list[dict]) -> list[dict]:
    evidence = "\n\n".join(
        f"[{i}] ({p['source']}: {p['focus']})\n{p['text']}" for i, p in enumerate(passages, start=1)
    )
    user = f"Evidence passages:\n\n{evidence}\n\nQuestion: {question}"
    return [{"role": "system", "content": GROUNDED_SYSTEM}, {"role": "user", "content": user}]


def llm_only_messages(question: str) -> list[dict]:
    return [{"role": "system", "content": LLM_ONLY_SYSTEM}, {"role": "user", "content": question}]
