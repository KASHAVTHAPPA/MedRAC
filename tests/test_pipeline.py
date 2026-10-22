from medrac_qa.evidence import lexical_support, validate_answer
from medrac_qa.pipeline import ABSTAIN_TEXT


def test_knowledge_base_search_ranks_relevant_first(kb, encoder):
    hits = kb.search(encoder.encode(["symptoms of anemia"])[0], k=2)
    assert hits[0].passage.focus == "anemia" and hits[0].score > hits[1].score


def test_validation_accepts_supported_answer(kb, encoder):
    hits = kb.search(encoder.encode(["anemia"])[0], k=2)
    v = validate_answer("Anemia causes fatigue and weakness [1].", [h.passage.text for h in hits],
                        kb.vectors(hits), encoder, 0.3, 0.34)
    assert v.passed


def test_validation_rejects_bad_citations(kb, encoder):
    hits = kb.search(encoder.encode(["anemia"])[0], k=2)
    texts, vecs = [h.passage.text for h in hits], kb.vectors(hits)
    assert "no citations" in validate_answer("Anemia causes fatigue.", texts, vecs, encoder, 0.3, 0.34).reason
    assert "do not exist" in validate_answer("Anemia causes fatigue [7].", texts, vecs, encoder, 0.3, 0.34).reason
    unsupported = validate_answer("Bananas cure everything quickly [1].", texts, vecs, encoder, 0.3, 0.34)
    assert not unsupported.passed and "not supported" in unsupported.reason
    assert not validate_answer("INSUFFICIENT_EVIDENCE", texts, vecs, encoder, 0.3, 0.34).passed


def test_lexical_support_handles_list_items():
    assert lexical_support("Pale skin", "Anemia causes fatigue weakness and pale skin.") == 1.0
    assert lexical_support("Bananas", "Anemia causes fatigue.") == 0.0


def test_emergency_skips_everything(make_pipeline):
    p, llm = make_pipeline()
    r = p.answer("I am having chest pain right now")
    assert r.route == "emergency" and llm.calls == 0


def test_rag_then_cache_hit(make_pipeline):
    p, llm = make_pipeline()
    first = p.answer("What are the symptoms of anemia?")
    assert first.route == "rag" and first.citations and llm.calls == 1
    second = p.answer("What are symptoms of anemia?")
    assert second.route == "cache" and llm.calls == 1      # no second LLM call
    assert second.answer == first.answer


def test_cache_not_reused_for_children(make_pipeline):
    p, llm = make_pipeline()
    p.answer("What are the symptoms of anemia?")
    r = p.answer("What are the symptoms of anemia in children?")
    assert r.route == "rag" and llm.calls == 2


def test_out_of_scope_abstains(make_pipeline):
    p, llm = make_pipeline()
    r = p.answer("Who won the football match yesterday?")
    assert r.route == "abstain" and r.answer == ABSTAIN_TEXT and llm.calls == 0


def test_unsupported_answer_falls_back_and_is_not_cached(make_pipeline):
    p, _ = make_pipeline(reply="Bananas cure everything quickly [1].")
    r = p.answer("What are the symptoms of anemia?")
    assert r.route == "fallback" and len(p.cache) == 0


def test_dosage_question_is_answered_but_not_cached(make_pipeline):
    p, _ = make_pipeline()
    r = p.answer("How much iron should I take for anemia symptoms?")
    assert r.route == "rag" and len(p.cache) == 0 and "not cacheable" in r.cache["reason"]


def test_rag_mode_never_uses_cache(make_pipeline):
    p, llm = make_pipeline()
    p.answer("What are the symptoms of anemia?", mode="rag")
    p.answer("What are the symptoms of anemia?", mode="rag")
    assert llm.calls == 2 and len(p.cache) == 0


def test_llm_only_mode(make_pipeline):
    p, llm = make_pipeline(reply="Anemia is a low red blood cell count.")
    r = p.answer("What is anemia?", mode="llm_only")
    assert r.route == "llm_only" and not r.citations and llm.calls == 1


def test_without_llm_returns_passages(make_pipeline):
    p, _ = make_pipeline()
    p.llm = None
    r = p.answer("What are the symptoms of anemia?")
    assert r.route == "retrieval_only" and r.citations
