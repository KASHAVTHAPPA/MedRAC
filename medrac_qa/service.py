"""Builds one ready-to-use pipeline from the settings (used by app.py and the scripts)."""

from __future__ import annotations

from pathlib import Path

from .cache import JsonFileBackend, MemoryBackend, RedisBackend, SemanticCache
from .config import Settings, load_dotenv
from .embedder import MiniLMEmbedder
from .knowledge import KnowledgeBase
from .llm import LLMClient
from .pipeline import MedRACPipeline


def build_pipeline(settings: Settings | None = None, cache_mode: str = "semantic") -> MedRACPipeline:
    load_dotenv()
    s = settings or Settings()
    kb = KnowledgeBase.load(s.index_dir)
    encoder = MiniLMEmbedder(s.model_dir)
    if s.redis_url and s.redis_token:
        # One key per knowledge-base version, so a rebuilt index starts a fresh cache.
        backend = RedisBackend(s.redis_url, s.redis_token, key=f"medrac:cache:{kb.version}")
    elif s.cache_file:
        backend = JsonFileBackend(Path(s.cache_file))
    else:
        backend = MemoryBackend()
    cache = SemanticCache(
        kb_version=kb.version,
        threshold=s.cache_threshold,
        ttl_days=s.cache_ttl_days,
        mode=cache_mode,
        backend=backend,
        passage_exists=lambda pid: kb.get(pid) is not None,
    )
    llm = LLMClient(
        s.llm_base_url, s.llm_api_key, s.llm_model, s.llm_timeout_s, s.llm_max_tokens, s.llm_reasoning_effort
    )
    return MedRACPipeline(s, kb, encoder, cache, llm)
