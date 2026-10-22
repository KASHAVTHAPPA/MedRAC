"""Rule-based safety screening, run before anything else.

Deliberately simple keyword rules instead of an LLM classifier: they are
predictable, testable and cannot be talked out of a decision. Two outputs:

1. emergency  -> the question describes an emergency happening now. We skip
                 RAG and the LLM entirely and return fixed guidance.
2. no_cache   -> the question may be answered, but its answer must never be
                 reused for someone else (personal details, doses,
                 time-sensitive questions).

An informational question about an emergency topic ("What are the symptoms
of a heart attack?") is *not* an emergency; it is answered normally, with
a short safety note added.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

EMERGENCY_MESSAGE = (
    "This sounds like it could be a medical emergency. Please call 112 (national "
    "emergency number) or 108 (ambulance) now, or go to the nearest emergency "
    "department. Do not wait for an online answer."
)
SELF_HARM_MESSAGE = (
    "You are not alone, and help is available right now. In India you can call "
    "Tele-MANAS on 14416 (free, 24x7) or 112 in an emergency. If you are in immediate "
    "danger, please contact someone you trust or go to the nearest hospital."
)
DISCLAIMER = (
    "This is general health information from NIH sources, not medical advice. "
    "Please consult a qualified doctor about your own situation."
)
EMERGENCY_TOPIC_NOTE = "If these signs are happening to someone right now, call 112 or 108 immediately."

_SELF_HARM = re.compile(
    r"\b(suicid\w*|kill (my|him|her)self|end (my|his|her) life|self[- ]?harm\w*|"
    r"want to die|hurt (my|him|her)self|cut(ting)? myself)\b",
    re.I,
)
# Acute danger signs ...
_DANGER = re.compile(
    r"\b(chest pain|crushing chest|can'?t breathe|cannot breathe|not breathing|"
    r"stopped breathing|trouble breathing|choking|unconscious|unresponsive|"
    r"passed out|fainted|seizure|convulsion|fitting|severe bleeding|"
    r"bleeding heavily|won'?t stop bleeding|face (is )?drooping|slurred speech|"
    r"overdos\w*|poison\w*|swallowed|throat (is )?(swelling|closing)|anaphyla\w*|"
    r"heart attack|stroke|blue lips|severe burn)\b",
    re.I,
)
# ... described as happening now, to the asker or someone near them.
_HAPPENING_NOW = re.compile(
    r"\b(i am|i'm|im|i have|i've|i was|i just|i think i|my \w+ (is|has|was|just)|"
    r"he is|she is|he's|she's|they are|someone is|right now|currently|suddenly|"
    r"help me|what should i do|what do i do|just took|just ate)\b",
    re.I,
)
_INFORMATIONAL = re.compile(r"^\s*(what|which|how (is|are|do|does|can)|why|when|who|is|are|can|does)\b", re.I)

_DOSE = re.compile(
    r"\b(dose|dosage|how much \w+ (should|can|do)|how many (tablets|pills|capsules|mg)|"
    r"\d+\s?(mg|mcg|ml|g|iu|units?)\b|maximum (dose|amount)|overdose)",
    re.I,
)
_PERSONAL = re.compile(
    r"\b(i am|i'm|i have|i've|my (son|daughter|baby|child|wife|husband|mother|father|"
    r"mom|dad|doctor|report|test|result|blood|sugar|bp)|i take|i was diagnosed|"
    r"\d+ ?(years?|yrs?|months?) old)\b",
    re.I,
)
_TIME_SENSITIVE = re.compile(r"\b(latest|newest|current(ly)?|this year|20\d\d|recent|new guideline|outbreak)\b", re.I)
_EMERGENCY_TOPIC = re.compile(r"\b(heart attack|stroke|anaphyla\w*|sepsis|overdose|poisoning|meningitis)\b", re.I)


@dataclass
class SafetyResult:
    emergency: bool = False
    message: str = ""
    no_cache_reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def cacheable(self) -> bool:
        return not self.emergency and not self.no_cache_reasons


def screen(question: str) -> SafetyResult:
    q = question.strip()
    result = SafetyResult()

    if _SELF_HARM.search(q):
        return SafetyResult(emergency=True, message=SELF_HARM_MESSAGE)

    danger = _DANGER.search(q)
    if danger and _HAPPENING_NOW.search(q):
        return SafetyResult(emergency=True, message=EMERGENCY_MESSAGE)
    if danger and not _INFORMATIONAL.search(q):
        # e.g. "chest pain and sweating" with no question form: treat as acute.
        return SafetyResult(emergency=True, message=EMERGENCY_MESSAGE)

    if _DOSE.search(q):
        result.no_cache_reasons.append("dosage question")
        result.notes.append("Medicine doses depend on the person; confirm any dose with a doctor or pharmacist.")
    if _PERSONAL.search(q):
        result.no_cache_reasons.append("personal details")
    if _TIME_SENSITIVE.search(q):
        result.no_cache_reasons.append("time-sensitive")
    if _EMERGENCY_TOPIC.search(q):
        result.notes.append(EMERGENCY_TOPIC_NOTE)
    return result
