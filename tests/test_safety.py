import pytest

from medrac_qa.safety import screen


@pytest.mark.parametrize(
    "q",
    [
        "My father has chest pain and is sweating right now",
        "I think I am having a stroke, my face is drooping",
        "my son swallowed rat poison what do i do",
        "chest pain and left arm numb",
        "I want to kill myself",
    ],
)
def test_emergencies_are_caught(q):
    assert screen(q).emergency


@pytest.mark.parametrize(
    "q",
    [
        "What are the symptoms of a heart attack?",
        "How is a stroke treated?",
        "What causes seizures?",
        "What is anemia?",
    ],
)
def test_informational_questions_are_answered(q):
    assert not screen(q).emergency


def test_emergency_topic_adds_note_but_stays_cacheable():
    r = screen("What are the warning signs of a heart attack?")
    assert r.notes and r.cacheable


@pytest.mark.parametrize(
    "q, reason",
    [
        ("How much paracetamol should I take for fever?", "dosage question"),
        ("Is 500 mg of metformin safe?", "dosage question"),
        ("I am 45 years old with diabetes, what should I eat?", "personal details"),
        ("What is the latest treatment for cancer?", "time-sensitive"),
    ],
)
def test_never_cache_rules(q, reason):
    r = screen(q)
    assert not r.cacheable and reason in r.no_cache_reasons


def test_plain_question_is_cacheable():
    assert screen("What are the symptoms of anemia?").cacheable
