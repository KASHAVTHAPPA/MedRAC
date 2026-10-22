"""What a question asks, about whom, and about what.

The semantic cache must not reuse an answer just because two questions
*sound* alike. "What are the symptoms of type 1 diabetes?" and "What are the
symptoms of type 2 diabetes?" have an embedding similarity above 0.95 but
different answers. This module extracts three things, by simple rules, that
must agree before a cached answer may be reused:

    intent      what is asked: symptoms, treatment, causes, ...
    qualifiers  who it is about and other clinically decisive words:
                children, pregnancy, elderly, negation, numbers (type 1/2)
    focus       the remaining content words (the disease, drug or test)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Order matters: the first matching intent wins.
INTENT_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("symptoms", re.compile(r"\b(symptom\w*|signs?|feel like|look like|warning signs)\b")),
    ("treatment", re.compile(r"\b(treat\w*|therap\w*|cure\w*|manag\w*|medication\w*|medicine\w*|drugs? for|remed\w*|get rid)\b")),
    ("prevention", re.compile(r"\b(prevent\w*|avoid\w*|reduce (the )?risk|protect\w*|vaccin\w*)\b")),
    ("diagnosis", re.compile(r"\b(diagnos\w*|test\w*|detect\w*|screen\w*|find out if|check for)\b")),
    ("inheritance", re.compile(r"\b(inherit\w*|genetic\w*|gene\w*|hereditary|run in famil\w*|passed down)\b")),
    ("outlook", re.compile(r"\b(outlook|prognosis|life expectancy|survival|recover\w*|go away|fatal|deadly)\b")),
    ("frequency", re.compile(r"\b(how common|how many people|frequency|prevalen\w*|incidence)\b")),
    ("causes", re.compile(r"\b(caus\w*|why do|why does|risk factors?|reasons? for|triggers?)\b")),
    ("complications", re.compile(r"\b(complication\w*|side effects?|lead to|long[- ]term effects?)\b")),
    ("research", re.compile(r"\b(research|clinical trials?|studies|new treatments?)\b")),
    ("information", re.compile(r".")),
]

QUALIFIERS: dict[str, re.Pattern[str]] = {
    "children": re.compile(r"\b(child\w*|kids?|infants?|bab(y|ies)|newborns?|toddlers?|paediatric|pediatric|teen\w*|adolescen\w*)\b"),
    "pregnancy": re.compile(r"\b(pregnan\w*|expecting|trimester|breastfeed\w*|breast[- ]feed\w*|lactat\w*|nursing mother)\b"),
    "elderly": re.compile(r"\b(elderly|older (adults?|people)|seniors?|aged|old age|over (60|65|70))\b"),
    "men": re.compile(r"\b(men|male|males|man)\b"),
    "women": re.compile(r"\b(women|female|females|woman)\b"),
    "negation": re.compile(r"\b(not|no|never|without|n't|cannot|non)\b"),
}

STOPWORDS = set(
    """a an the of in on for to and or with without about is are was were be been being do does did can could
    should would will may might must what which who whom whose when where why how this that these those it its
    i me my we our you your he she they them his her their there here have has had having get gets getting
    from by as at into than then so if any all some more most other such only own same too very just also
    tell explain know please describe give information info like common usual usually typical possible
    available best main different kinds types type way ways options option happen happens someone person
    people affect affects affected mean means called s seen overview done used long term need occur
    occurs condition""".split()
)
_INTENT_WORDS = re.compile(
    r"\b(symptom\w*|signs?|treat\w*|therap\w*|cure\w*|manag\w*|medication\w*|medicine\w*|remed\w*|"
    r"prevent\w*|avoid\w*|diagnos\w*|test\w*|detect\w*|screen\w*|inherit\w*|genetic\w*|gene\w*|hereditary|"
    r"outlook|prognosis|caus\w*|risk|factors?|complication\w*|research|trials?|studies|frequency|"
    r"common|many|expectancy|survival|prevalen\w*|famil\w*|passed|down|run)\b"
)
# Digits (and subtype codes such as "1B" or "2F") anywhere; roman numerals only
# after "type/stage/grade/..." so the pronoun "I" is not read as the number one.
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?[a-z]?\b")
_ROMAN_AFTER = re.compile(r"\b(?:type|stage|grade|class|phase)\s+(i{1,3}|iv|v)\b")
_ROMAN = {"i": "1", "ii": "2", "iii": "3", "iv": "4", "v": "5"}
_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class QueryProfile:
    intent: str
    qualifiers: frozenset[str]
    numbers: frozenset[str]
    focus: frozenset[str]

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "qualifiers": sorted(self.qualifiers),
            "numbers": sorted(self.numbers),
            "focus": sorted(self.focus),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "QueryProfile":
        return cls(d["intent"], frozenset(d["qualifiers"]), frozenset(d["numbers"]), frozenset(d["focus"]))


def _stem(word: str) -> str:
    """Very light plural stemming so 'diseases' matches 'disease'."""
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def analyze(question: str) -> QueryProfile:
    q = question.lower().replace("’", "'")
    intent = next(name for name, rule in INTENT_RULES if rule.search(q))
    qualifiers = frozenset(name for name, rule in QUALIFIERS.items() if rule.search(q))
    numbers = frozenset(_NUMBER.findall(q)) | frozenset(_ROMAN[r] for r in _ROMAN_AFTER.findall(q))

    rest = _NUMBER.sub(" ", _ROMAN_AFTER.sub(" ", _INTENT_WORDS.sub(" ", q)))
    for rule in QUALIFIERS.values():
        rest = rule.sub(" ", rest)
    focus = frozenset(_stem(w) for w in _WORD.findall(rest) if w not in STOPWORDS and len(w) > 1)
    return QueryProfile(intent, qualifiers, numbers, focus)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def compatible(new: QueryProfile, cached: QueryProfile, min_focus_overlap: float = 0.75) -> tuple[bool, str]:
    """Can an answer written for `cached` be reused for `new`? Returns (ok, reason)."""
    if new.intent != cached.intent:
        return False, f"different intent ({new.intent} vs {cached.intent})"
    if new.qualifiers != cached.qualifiers:
        diff = sorted(new.qualifiers ^ cached.qualifiers)
        return False, f"different clinical context ({', '.join(diff)})"
    if new.numbers != cached.numbers:
        return False, "different numbers or types"
    overlap = jaccard(new.focus, cached.focus)
    if overlap < min_focus_overlap:
        return False, f"different topic words (overlap {overlap:.2f})"
    return True, "compatible"
