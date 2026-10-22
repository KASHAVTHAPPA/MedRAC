"""Shared Redis cache: two server instances, one fake Redis."""

from medrac_qa.cache import RedisBackend, SemanticCache
from medrac_qa.query_analysis import analyze


class FakeRedis:
    """Implements the few list commands the backend uses, like Upstash's /pipeline."""

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.values: dict[str, str] = {}
        self.down = False
        self.requests = 0

    def __call__(self, commands: list[list]) -> list:
        self.requests += 1
        if self.down:
            raise ConnectionError("redis is down")
        out = []
        for cmd, key, *args in commands:
            if cmd == "GET":
                out.append(self.values.get(key))
                continue
            if cmd == "INCR":
                self.values[key] = str(int(self.values.get(key) or 0) + 1)
                out.append(int(self.values[key]))
                continue
            items = self.lists.setdefault(key, [])
            if cmd == "LRANGE":
                start, end = int(args[0]), int(args[1])
                out.append(items[start:] if end == -1 else items[start : end + 1])
            elif cmd == "RPUSH":
                items.extend(args)
                out.append(len(items))
            elif cmd == "DEL":
                out.append(int(bool(self.lists.pop(key, None))))
        return out


def _instance(redis: FakeRedis, sync_interval_s: float = 30.0) -> SemanticCache:
    backend = RedisBackend("https://x", "t", transport=redis)
    return SemanticCache("v1", threshold=0.8, backend=backend, sync_interval_s=sync_interval_s)


def _store(cache, encoder, q):
    cache.store(q, encoder.encode([q])[0], analyze(q), "shared answer", [{"id": "P0_c0", "n": 1}])


def _lookup(cache, encoder, q):
    return cache.lookup(q, encoder.encode([q])[0], analyze(q))


def test_answer_stored_by_one_instance_is_reused_by_another(encoder):
    redis = FakeRedis()
    a, b = _instance(redis), _instance(redis)
    _store(a, encoder, "What are the symptoms of anemia?")
    d = _lookup(b, encoder, "What are the symptoms of anemia?")
    assert d.hit and d.entry.answer == "shared answer"


def test_new_instance_loads_existing_entries(encoder):
    redis = FakeRedis()
    _store(_instance(redis), encoder, "What is anemia?")
    late = _instance(redis)  # e.g. a cold start after the first answer
    assert len(late) == 1 and _lookup(late, encoder, "What is anemia?").hit


def test_own_entries_are_not_duplicated(encoder):
    redis = FakeRedis()
    a = _instance(redis)
    _store(a, encoder, "What is anemia?")
    _lookup(a, encoder, "What is anemia?")   # sync pulls our own entry back
    _lookup(a, encoder, "What is anemia?")
    assert len(a) == 1


def test_clear_on_one_instance_resets_the_others(encoder):
    redis = FakeRedis()
    a, b = _instance(redis), _instance(redis, sync_interval_s=0)  # b syncs before every lookup
    _store(a, encoder, "What is anemia?")
    _store(a, encoder, "How is glaucoma diagnosed?")
    assert _lookup(b, encoder, "What is anemia?").hit
    a.clear()
    # The new list grows longer than what b had seen; the generation still tells b to reset.
    for q in ("What causes gout?", "What is psoriasis?", "What is asthma?"):
        _store(a, encoder, q)
    assert not _lookup(b, encoder, "What is anemia?").hit
    assert len(b) == 3


def test_redis_outage_does_not_break_answering(encoder):
    redis = FakeRedis()
    a = _instance(redis)
    redis.down = True
    _store(a, encoder, "What is anemia?")           # save fails quietly
    d = _lookup(a, encoder, "What is anemia?")     # still served from local copy
    assert d.hit
    assert not _lookup(a, encoder, "How is glaucoma diagnosed?").hit   # failed sync = plain miss
    assert a.backend.errors == 2


def test_local_hit_needs_no_redis_call_and_miss_needs_one(encoder):
    redis = FakeRedis()
    a = _instance(redis)
    _store(a, encoder, "What is anemia?")
    before = redis.requests
    assert _lookup(a, encoder, "What is anemia?").hit
    assert redis.requests == before                  # served from the local copy
    assert not _lookup(a, encoder, "How is glaucoma diagnosed?").hit
    assert redis.requests == before + 1              # miss: one sync to check other instances
