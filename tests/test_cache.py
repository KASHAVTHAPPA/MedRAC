from medrac_qa.cache import DAY, JsonFileBackend, SemanticCache
from medrac_qa.query_analysis import analyze

CITES = [{"id": "P0_c0", "n": 1}]


def _cache(encoder, **kw):
    kw.setdefault("threshold", 0.8)
    return SemanticCache("v1", **kw), encoder


def _store(cache, encoder, q, answer="cached answer", citations=CITES):
    cache.store(q, encoder.encode([q])[0], analyze(q), answer, citations)


def _lookup(cache, encoder, q):
    return cache.lookup(q, encoder.encode([q])[0], analyze(q))


def test_empty_cache_misses(encoder):
    cache, enc = _cache(encoder)
    assert not _lookup(cache, enc, "What is anemia?").hit


def test_same_question_hits_and_counts(encoder):
    cache, enc = _cache(encoder)
    _store(cache, enc, "What are the symptoms of anemia?")
    d = _lookup(cache, enc, "What are the symptoms of anemia?")
    assert d.hit and d.entry.answer == "cached answer"
    assert cache.summary()["hits"] == 1 and d.entry.hits == 1


def test_low_similarity_misses(encoder):
    cache, enc = _cache(encoder)
    _store(cache, enc, "What are the symptoms of anemia?")
    d = _lookup(cache, enc, "How is glaucoma diagnosed?")
    assert not d.hit and "below" in d.reason


def test_context_gate_blocks_different_population(encoder):
    cache, enc = _cache(encoder, threshold=0.5)
    _store(cache, enc, "What are the symptoms of anemia?")
    d = _lookup(cache, enc, "What are the symptoms of anemia in children?")
    assert d.similarity >= 0.5            # similar enough for a naive cache ...
    assert not d.hit and "children" in d.reason   # ... but rejected by the gate


def test_naive_mode_would_reuse_it(encoder):
    cache, enc = _cache(encoder, threshold=0.5, mode="naive")
    _store(cache, enc, "What are the symptoms of anemia?")
    assert _lookup(cache, enc, "What are the symptoms of anemia in children?").hit


def test_exact_mode_needs_identical_text(encoder):
    cache, enc = _cache(encoder, mode="exact")
    _store(cache, enc, "What are the symptoms of anemia?")
    assert _lookup(cache, enc, "what are the symptoms of ANEMIA").hit
    assert not _lookup(cache, enc, "What signs does anemia cause?").hit


def test_stale_entries_are_not_reused(encoder):
    now = [1_000_000.0]
    cache = SemanticCache("v1", threshold=0.8, ttl_days=30, clock=lambda: now[0])
    _store(cache, encoder, "What is anemia?")
    now[0] += 31 * DAY
    d = _lookup(cache, encoder, "What is anemia?")
    assert not d.hit and "out of date" in d.reason


def test_new_knowledge_base_version_invalidates(encoder):
    cache = SemanticCache("v1", threshold=0.8)
    _store(cache, encoder, "What is anemia?")
    cache.kb_version = "v2"
    assert not _lookup(cache, encoder, "What is anemia?").hit


def test_missing_evidence_invalidates(encoder):
    cache = SemanticCache("v1", threshold=0.8, passage_exists=lambda pid: False)
    _store(cache, encoder, "What is anemia?")
    d = _lookup(cache, encoder, "What is anemia?")
    assert not d.hit and "evidence" in d.reason


def test_json_backend_persists(tmp_path, encoder):
    path = tmp_path / "cache.jsonl"
    cache = SemanticCache("v1", threshold=0.8, backend=JsonFileBackend(path))
    _store(cache, encoder, "What is anemia?")
    reloaded = SemanticCache("v1", threshold=0.8, backend=JsonFileBackend(path))
    assert len(reloaded) == 1 and _lookup(reloaded, encoder, "What is anemia?").hit
    reloaded.clear()
    assert not path.exists()
