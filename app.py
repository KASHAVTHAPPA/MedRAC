"""MedRAC-QA web API (FastAPI). Vercel detects this file automatically.

    POST /api/ask          {"question": "...", "mode": "full" | "rag" | "llm_only"}
    GET  /api/health       index, model and cache status
    GET  /api/cache        cache statistics and recent entries
    POST /api/cache/clear  empty the cache (demo use)

The web page itself is plain HTML/JS in public/, served by Vercel's CDN
(and by FastAPI's StaticFiles when running locally with uvicorn).

Run locally:   uvicorn app:app --reload   then open http://127.0.0.1:8000
"""

from __future__ import annotations

import os
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from medrac_qa import __version__
from medrac_qa.pipeline import MODES
from medrac_qa.service import build_pipeline

PUBLIC = Path(__file__).resolve().parent / "public"
RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MIN", "20"))

app = FastAPI(title="MedRAC-QA", version=__version__, docs_url="/api/docs", openapi_url="/api/openapi.json")

# Loaded once per server instance (about 1 s: index + ONNX model), then reused.
pipeline = build_pipeline()
_recent: dict[str, deque] = defaultdict(deque)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    mode: str = "full"


def _rate_limited(ip: str) -> bool:
    """At most RATE_LIMIT questions per minute per IP (per server instance)."""
    now = time.time()
    q = _recent[ip]
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT:
        return True
    q.append(now)
    return False


@app.post("/api/ask")
def ask(body: AskRequest, request: Request) -> dict:
    if body.mode not in MODES:
        raise HTTPException(400, f"mode must be one of {MODES}")
    ip = request.headers.get("x-forwarded-for", request.client.host if request.client else "?").split(",")[0]
    if _rate_limited(ip):
        raise HTTPException(429, "Too many questions. Please wait a minute.")
    return pipeline.answer(body.question.strip(), body.mode).to_dict()


@app.get("/api/health")
def health() -> dict:
    s = pipeline.s
    return {
        "status": "ok",
        "version": __version__,
        "passages": len(pipeline.kb),
        "kb_version": pipeline.kb.version,
        "llm_model": s.llm_model if pipeline.llm else None,
        "llm_configured": pipeline.llm is not None,
        "settings": {
            "top_k": s.top_k,
            "cache_threshold_tau": s.cache_threshold,
            "evidence_threshold_gamma": s.evidence_threshold,
            "support_threshold_delta": s.support_threshold,
        },
        "cache": pipeline.cache.summary(),
    }


@app.get("/api/cache")
def cache_entries() -> dict:
    entries = [
        {"question": e.question, "hits": e.hits, "intent": e.profile["intent"], "created_at": e.created_at}
        for e in reversed(pipeline.cache.entries[-50:])
    ]
    return {**pipeline.cache.summary(), "recent": entries}


@app.post("/api/cache/clear")
def cache_clear() -> dict:
    pipeline.cache.clear()
    return pipeline.cache.summary()


# Local development only: on Vercel, files in public/ are served by the CDN
# before a request ever reaches this function.
@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(PUBLIC / "index.html")


@app.get("/{name}", include_in_schema=False)
def static_file(name: str) -> FileResponse:
    path = (PUBLIC / name).resolve()
    if path.parent != PUBLIC or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)
