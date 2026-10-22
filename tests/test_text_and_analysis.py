from medrac_qa.query_analysis import analyze, compatible
from medrac_qa.text import chunk_words, citations_in, normalize_question, split_sentences, strip_citations


def test_chunk_words_overlap_and_coverage():
    text = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_words(text, size=180, overlap=30)
    assert [len(c.split()) for c in chunks] == [180, 180, 100]
    assert chunks[1].split()[0] == "w150"          # 30-word overlap
    assert chunks[-1].split()[-1] == "w399"        # nothing lost at the end


def test_short_text_is_one_chunk():
    assert chunk_words("just a few words") == ["just a few words"]
    assert chunk_words("") == []


def test_sentences_and_citations():
    s = split_sentences("Anemia causes fatigue [1]. It is common [2][3].\n- Pale skin [1]")
    assert s == ["Anemia causes fatigue [1].", "It is common [2][3].", "Pale skin [1]"]
    assert citations_in(s[1]) == [2, 3]
    assert strip_citations(s[1]) == "It is common ."


def test_normalize_question():
    assert normalize_question("  What is  Anemia? ") == "what is anemia"


def test_intent_detection():
    assert analyze("What are the symptoms of anemia?").intent == "symptoms"
    assert analyze("How is asthma treated?").intent == "treatment"
    assert analyze("What causes high blood pressure?").intent == "causes"
    assert analyze("How is glaucoma diagnosed?").intent == "diagnosis"
    assert analyze("What is glaucoma?").intent == "information"


def test_qualifiers_numbers_and_focus():
    p = analyze("What are the symptoms of type 2 diabetes in children?")
    assert p.qualifiers == {"children"}
    assert p.numbers == {"2"}
    assert p.focus == {"diabete"}  # "diabetes" -> light plural stem
    assert analyze("Is type II diabetes inherited?").numbers == {"2"}
    assert analyze("What should I do about asthma?").numbers == frozenset()  # "I" is not a number


def test_paraphrases_are_compatible():
    a = analyze("What are the symptoms of anemia?")
    b = analyze("What signs and symptoms does anemia cause?")
    assert compatible(a, b)[0]


def test_clinically_different_questions_are_not_compatible():
    base = analyze("What are the symptoms of diabetes?")
    for other, reason in [
        ("What are the symptoms of diabetes in children?", "clinical context"),
        ("What are the symptoms of diabetes during pregnancy?", "clinical context"),
        ("How is diabetes treated?", "intent"),
        ("What are the symptoms of asthma?", "topic"),
    ]:
        ok, why = compatible(analyze(other), base)
        assert not ok and reason in why, other
    ok, why = compatible(analyze("What are the symptoms of type 1 diabetes?"), analyze("What are the symptoms of type 2 diabetes?"))
    assert not ok and "numbers" in why
