"""The MedRAC-QA algorithm: safety -> CAG -> RAG -> LLM -> validation -> cache.

    1. screen the question             (safety.screen)
    2. if emergency: return fixed guidance, stop
    3. embed the question               (MiniLM)
    4. CAG: look up the semantic cache  -> validated hit? return it, stop
    5. RAG: retrieve top-k passages
    6. evidence gate: best score < gamma? abstain, stop
    7. LLM writes a cited answer from the passages
    8. validate citations and support   -> failed? safe fallback, stop
    9. store the validated answer in the cache (if the question is cacheable)

Modes let the same code produce the baselines for the experiments:
    "full"      everything above                       (proposed system, B6)
    "rag"       steps 1-3 and 5-8, no cache             (RAG + LLM, B2)
    "llm_only"  step 1 and the LLM with no evidence    (B1)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import safety
from .cache import SemanticCache
from .config import Settings
from .embedder import TextEncoder
from .evidence import sufficiency, validate_answer
from .knowledge import Hit, KnowledgeBase
from .llm import LLMClient, LLMError
from .prompts import grounded_messages, llm_only_messages
from .query_analysis import analyze

MODES = ("full", "rag", "llm_only")

ABSTAIN_TEXT = (
    "I could not find reliable information about this in the NIH knowledge base, so I "
    "would rather not guess. Please ask a doctor, or rephrase the question if it is "
    "about a specific disease, test or treatment."
)
FALLBACK_TEXT = (
    "I could not write an answer that is fully supported by the sources, so here are "
    "the most relevant passages from NIH instead."
)
NO_LLM_TEXT = "The language model is not configured, so here are the most relevant passages from NIH."


@dataclass
class QAResult:
    question: str
    route: str                 # emergency | cache | rag | abstain | fallback | llm_only | retrieval_only | error
    answer: str
    citations: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    cache: dict | None = None
    evidence_score: float | None = None
    validation: dict | None = None
    profile: dict | None = None
    timings_ms: dict = field(default_factory=dict)
    model: str | None = None
    disclaimer: str = safety.DISCLAIMER

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def _citation(i: int, hit: Hit) -> dict:
    p = hit.passage
    return {
        "n": i,
        "id": p.id,
        "focus": p.focus,
        "source": p.source,
        "url": p.url,
        "question": p.question,
        "text": p.text,
        "score": round(hit.score, 4),
    }


class MedRACPipeline:
    def __init__(
        self,
        settings: Settings,
        kb: KnowledgeBase,
        encoder: TextEncoder,
        cache: SemanticCache,
        llm: LLMClient | None,
    ) -> None:
        self.s = settings
        self.kb = kb
        self.encoder = encoder
        self.cache = cache
        self.llm = llm if settings.llm_configured else None

    def answer(self, question: str, mode: str = "full") -> QAResult:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        t: dict[str, float] = {}
        start = time.perf_counter()

        def lap(name: str, since: float) -> float:
            now = time.perf_counter()
            t[name] = round((now - since) * 1000, 1)
            return now

        # 1-2. Safety screening
        check = safety.screen(question)
        mark = lap("safety", start)
        if check.emergency:
            t["total"] = t["safety"]
            return QAResult(question, "emergency", check.message, timings_ms=t)

        profile = analyze(question)

        if mode == "llm_only":
            return self._llm_only(question, check.notes, profile.to_dict(), t, start)

        # 3. Embed the question once; the vector is reused by cache and retrieval.
        q_vec = self.encoder.encode([question])[0]
        mark = lap("embed", mark)

        # 4. CAG: validated semantic cache
        cache_info = None
        if mode == "full" and not check.cacheable:
            cache_info = {"hit": False, "reason": f"not cacheable ({', '.join(check.no_cache_reasons)})", "checks": {}}
        elif mode == "full":
            decision = self.cache.lookup(question, q_vec, profile)
            mark = lap("cache_lookup", mark)
            cache_info = decision.to_dict()
            if decision.hit and decision.entry:
                t["total"] = round((time.perf_counter() - start) * 1000, 1)
                return QAResult(
                    question, "cache", decision.entry.answer, decision.entry.citations, check.notes,
                    cache_info, profile=profile.to_dict(), timings_ms=t,
                )

        # 5. RAG: retrieve evidence
        hits = self.kb.search(q_vec, self.s.top_k)
        mark = lap("retrieve", mark)
        citations = [_citation(i, h) for i, h in enumerate(hits, start=1)]

        # 6. Evidence sufficiency gate
        enough, best = sufficiency(hits, self.s.evidence_threshold)
        if not enough:
            t["total"] = round((time.perf_counter() - start) * 1000, 1)
            return QAResult(question, "abstain", ABSTAIN_TEXT, [], check.notes, cache_info, best,
                            profile=profile.to_dict(), timings_ms=t)

        if self.llm is None:
            t["total"] = round((time.perf_counter() - start) * 1000, 1)
            return QAResult(question, "retrieval_only", NO_LLM_TEXT, citations[:3], check.notes, cache_info, best,
                            profile=profile.to_dict(), timings_ms=t)

        # 7. Grounded generation
        try:
            reply = self.llm.chat(grounded_messages(question, citations))
        except LLMError as e:
            t["total"] = round((time.perf_counter() - start) * 1000, 1)
            return QAResult(question, "error", f"{NO_LLM_TEXT} ({e})", citations[:3], check.notes, cache_info, best,
                            profile=profile.to_dict(), timings_ms=t)
        mark = lap("generate", mark)

        # 8. Validate citations and support
        validation = validate_answer(reply.text, [h.passage.text for h in hits], self.kb.vectors(hits), self.encoder,
                                     self.s.support_threshold, self.s.max_unsupported_rate)
        mark = lap("validate", mark)
        if not validation.passed:
            t["total"] = round((time.perf_counter() - start) * 1000, 1)
            return QAResult(question, "fallback", FALLBACK_TEXT, citations[:3], check.notes, cache_info, best,
                            validation.to_dict(), profile.to_dict(), t, reply.model)

        # 9. Admit the validated answer to the cache
        used = sorted({c["n"] for c in citations if f"[{c['n']}]" in reply.text})
        shown = [c for c in citations if c["n"] in used]
        if mode == "full" and check.cacheable:
            self.cache.store(question, q_vec, profile, reply.text, shown)
            lap("cache_store", mark)

        t["total"] = round((time.perf_counter() - start) * 1000, 1)
        return QAResult(question, "rag", reply.text, shown, check.notes, cache_info, best,
                        validation.to_dict(), profile.to_dict(), t, reply.model)

    def _llm_only(self, question: str, notes: list[str], profile: dict, t: dict, start: float) -> QAResult:
        if self.llm is None:
            return QAResult(question, "error", "The language model is not configured.", notes=notes, timings_ms=t)
        try:
            reply = self.llm.chat(llm_only_messages(question))
        except LLMError as e:
            return QAResult(question, "error", str(e), notes=notes, timings_ms=t)
        t["generate"] = round(reply.latency_ms, 1)
        t["total"] = round((time.perf_counter() - start) * 1000, 1)
        return QAResult(question, "llm_only", reply.text, [], notes, profile=profile, timings_ms=t, model=reply.model)
