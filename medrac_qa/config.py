"""All tunable settings in one place, read from environment variables.

Every value has a sensible default, so the app runs with no configuration
except an LLM API key. The thresholds (tau, gamma, delta) were chosen on the
development split by the scripts in experiments/ (see docs/RESULTS.md).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Read KEY=value lines from .env for local runs. Real environment
    variables (e.g. those set in the Vercel dashboard) always win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _float(name: str, default: float) -> float:
    return float(_env(name, str(default)))


def _int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


@dataclass
class Settings:
    # ---- Language model (any OpenAI-compatible endpoint) ----
    # Gemini (hosted, free tier): https://generativelanguage.googleapis.com/v1beta/openai + gemini-3.5-flash-lite
    # Groq (hosted, free tier):   https://api.groq.com/openai/v1  + llama-3.1-8b-instant
    # Ollama (local, offline):    http://localhost:11434/v1       + qwen2.5:3b
    llm_base_url: str = field(
        default_factory=lambda: _env("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
    )
    llm_api_key: str = field(default_factory=lambda: _env("LLM_API_KEY", ""))
    llm_model: str = field(default_factory=lambda: _env("LLM_MODEL", "gemini-3.5-flash-lite"))
    llm_timeout_s: float = field(default_factory=lambda: _float("LLM_TIMEOUT_S", 30.0))
    llm_max_tokens: int = field(default_factory=lambda: _int("LLM_MAX_TOKENS", 400))
    # Optional: "minimal" / "low" for models that think before answering (Gemini 3.x)
    llm_reasoning_effort: str = field(default_factory=lambda: _env("LLM_REASONING_EFFORT", ""))

    # ---- RAG ----
    top_k: int = field(default_factory=lambda: _int("TOP_K", 5))
    # gamma: minimum similarity of the best passage before we try to answer
    evidence_threshold: float = field(default_factory=lambda: _float("EVIDENCE_THRESHOLD", 0.40))
    # delta: minimum similarity between an answer sentence and its cited passage
    support_threshold: float = field(default_factory=lambda: _float("SUPPORT_THRESHOLD", 0.40))
    # an answer is rejected if more than this fraction of its sentences is unsupported
    max_unsupported_rate: float = field(default_factory=lambda: _float("MAX_UNSUPPORTED_RATE", 0.34))

    # ---- CAG (semantic answer cache) ----
    # tau: minimum cosine similarity between two questions for a cache hit
    cache_threshold: float = field(default_factory=lambda: _float("CACHE_THRESHOLD", 0.85))
    cache_ttl_days: float = field(default_factory=lambda: _float("CACHE_TTL_DAYS", 30.0))
    # Where cached answers live, in order of preference:
    #   1. shared Redis (Upstash REST) when its URL + token are set: all server
    #      instances share one cache. Vercel's Upstash integration sets either
    #      the UPSTASH_REDIS_REST_* or the KV_REST_API_* names automatically.
    #   2. a JSON-lines file when CACHE_FILE is set (local runs)
    #   3. memory of this process
    redis_url: str = field(
        default_factory=lambda: _env("UPSTASH_REDIS_REST_URL", _env("KV_REST_API_URL", ""))
    )
    redis_token: str = field(
        default_factory=lambda: _env("UPSTASH_REDIS_REST_TOKEN", _env("KV_REST_API_TOKEN", ""))
    )
    cache_file: str = field(default_factory=lambda: _env("CACHE_FILE", ""))

    # ---- Files ----
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR", str(ROOT / "data" / "index"))))
    model_dir: Path = field(default_factory=lambda: Path(_env("MODEL_DIR", str(ROOT / "models" / "minilm"))))

    @property
    def llm_configured(self) -> bool:
        """Local servers such as Ollama need no key; hosted APIs do."""
        local = "localhost" in self.llm_base_url or "127.0.0.1" in self.llm_base_url
        return local or bool(self.llm_api_key)
