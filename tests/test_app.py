"""API tests against the real MedQuAD index and MiniLM model, with a fake LLM."""

import pytest
from fastapi.testclient import TestClient

from tests.conftest import FakeLLM


@pytest.fixture(scope="module")
def client():
    import app as app_module

    app_module.pipeline.llm = FakeLLM("Anemia is a condition in which the body lacks healthy red blood cells [1].")
    app_module.pipeline.cache.clear()
    return TestClient(app_module.app)


def test_health(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["passages"] > 20000


def test_real_retrieval_finds_anemia(client):
    r = client.post("/api/ask", json={"question": "What is anemia?", "mode": "rag"}).json()
    assert r["route"] == "rag"
    assert "anemia" in r["citations"][0]["focus"].lower()
    assert r["evidence_score"] > 0.6


def test_paraphrase_is_served_from_cache(client):
    client.post("/api/ask", json={"question": "What is anemia?"})
    r = client.post("/api/ask", json={"question": "Can you tell me what anemia is?"}).json()
    assert r["route"] == "cache" and r["cache"]["similarity"] >= 0.9


def test_out_of_scope_abstains(client):
    r = client.post("/api/ask", json={"question": "Who won the cricket world cup in 2011?"}).json()
    assert r["route"] == "abstain"


def test_input_validation(client):
    assert client.post("/api/ask", json={"question": "x"}).status_code == 422
    assert client.post("/api/ask", json={"question": "What is anemia?", "mode": "bad"}).status_code == 400


def test_static_files_and_traversal(client):
    assert client.get("/").status_code == 200
    assert client.get("/app.js").status_code == 200
    assert client.get("/..%2Fapp.py").status_code == 404
