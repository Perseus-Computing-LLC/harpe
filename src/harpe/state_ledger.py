"""StateLedger: question-gated pairing of a first-person claim with its later denial (or the reverse).

Round 8 candidate for BEAM's contradiction_resolution ability. The detector reads only two things: the dialogue
units it is given (speaker-tagged turn text with a date) and the question text at recall time. It never sees an
ability label, a rubric, a gold answer or a score; the ``tests/test_state_ledger.py`` source scan asserts that
those words do not appear in this module.

Why this exists. ``find_conflicts`` (``--surface-conflicts``, Round 3) inserted a bracketed note after the denying
sentence inside the retrieved context: ``[possible conflict: on <date>, user said "..."]``. On BEAM 1M the
contradiction ability fell 1.25 points (95% CI -2.5 to -0.2). The note reads as an instruction, and the reader
answered more tersely instead of pointing out the disagreement, which is what BEAM's rubric rewards. StateLedger
changes three things: (1) it only fires when the question itself asks about a first-person past action or state,
so the other nine abilities get byte-identical prompts; (2) it guarantees both sides of the pair reach the reader
even when retrieval selected only one; (3) the header it emits quotes the two statements with their dates and
contains no instruction text at all - no "conflict", no "note", no "possible", no brackets.

Round 9 opens the verb vocabulary. Round 8's miss analysis found 30 of the 39 unpaired contradiction questions
used a verb outside the 17-word list (``deployed``, ``migrated``, ``upgraded`` ...). AFFIRM now accepts
``I've <participle>`` for any participle-shaped word and DENY accepts ``I've never <word>`` / ``haven't <word>``,
both tolerating one adverb (``actually``, ``really``, ``ever`` ...). The wider net is paid for at pairing time: an
affirmation and a denial pair only when they share the same (stemmed) verb or at least two specific content words,
so generic overlap (``project``, ``work``, ``during``) cannot pair two unrelated statements.

Everything here is lexical and deterministic. Freeze the file by SHA-256 before any paid inference.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

_VERBS = (
    r"completed|implemented|created|fixed|written|set up|developed|built|configured|tested|solved|attended|met|"
    r"signed|updated|purchased|taken"
)
_SUBJ = r"(?:i(?:'ve| have)?|we(?:'ve| have)?)"
# One optional adverb between the subject (or the negation) and the verb: "I've actually finished", "haven't really".
_ADV = r"(?:actually\s+|really\s+|ever\s+|already\s+|recently\s+|just\s+|finally\s+|personally\s+)?"
# A participle-shaped word: the open-vocabulary affirmation verb. "not" is excluded so "I've not done" is a denial;
# the trailing (?!') keeps "haven" out of "I haven't".
_PARTICIPLE = r"(?!not\b)(\w+(?:ed|en|t|wn|ne|de|lt))\b(?!')"

AFFIRM = (
    re.compile(rf"\b{_SUBJ}\s+{_ADV}{_PARTICIPLE}", re.IGNORECASE),  # I've <participle>; group 1 = verb
    re.compile(rf"\b(?:i|we)\s+(?:always|usually)\b", re.IGNORECASE),
    re.compile(rf"\byes,?\s+{_SUBJ}\b", re.IGNORECASE),
)
DENY = (
    re.compile(rf"\b{_SUBJ}\s+(?:never|not yet|not)\s+{_ADV}(\w+)\b", re.IGNORECASE),  # I've never <word>; group 1
    re.compile(rf"\b(?:haven't|have not|didn't|did not|never)\s+{_ADV}(\w+)\b", re.IGNORECASE),  # group 1 = verb
    re.compile(rf"\b(?:i|we)\s+(?:never|rarely)\b", re.IGNORECASE),
    re.compile(r"\b(?:refused to|decided against|avoided)\b", re.IGNORECASE),
)
# The question gate. A header is only considered for questions that ask about the user's own past action or state.
QUESTION_GATE = re.compile(r"^(?:have (?:i|you)|has [a-z]+|do i|did i|how experienced am i)\b", re.IGNORECASE)
MIN_SHARED = 2  # content words the question, the claim and the denial must all share
# Words that two statements can share without being about the same thing; a pair needs the same verb or two
# shared words outside this set.
GENERIC = frozenset("project any this that some really actually ever before during yet thing things work "
                    "time way one also just still even much many more".split())
_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ied", "ers", "er", "ed", "es", "s")


def stem(word: str) -> str:
    """Crude suffix stemmer, enough to match 'deployed' with 'deploying' and 'deploy'."""
    for s in _SUFFIXES:
        if word.endswith(s) and len(word) - len(s) >= 3:
            return word[: -len(s)]
    return word


def _verb_of(sentence: str, patterns: Sequence["re.Pattern[str]"]) -> str:
    for p in patterns:
        m = p.search(sentence)
        if m and m.groups() and m.group(1):
            return stem(m.group(1).lower())
    return ""

_STOP = frozenset("the a an in on at of to for with by from is was are were and or but so that it this have has had "
                  "been be".split())
_WORD = re.compile(r"\b[a-z]{3,}\b")
_ROLE = re.compile(r"^\s*(user|assistant|system)\s*:\s*", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=\S)")


def content_words(text: str) -> frozenset:
    return frozenset(w for w in _WORD.findall(text.lower()) if w not in _STOP)


class Claim(BaseModel):
    """One user sentence that affirms (+1) or denies (-1) a first-person action or state."""

    unit_id: str
    date: str = ""
    polarity: int = Field(ge=-1, le=1)
    text: str
    words: frozenset = Field(default_factory=frozenset)
    verb: str = ""  # stemmed verb the pattern matched, "" when the pattern has no verb group

    model_config = {"arbitrary_types_allowed": True, "frozen": True}


class StatePair(BaseModel):
    affirm: Claim
    deny: Claim
    shared: Tuple[str, ...]


def _user_sentences(text: str) -> Iterable[str]:
    """Sentences of a unit spoken by the user. A role tag opens a speaker's span; untagged lines continue the
    current speaker; a unit with no tags at all is user speech."""
    speaker, span = "user", []
    for line in text.splitlines():
        m = _ROLE.match(line)
        if m:
            if speaker == "user":
                yield from _SENTENCE_END.split(" ".join(span))
            speaker, span = m.group(1).lower(), [line[m.end():].strip()]
        elif line.strip():
            span.append(line.strip())
    if speaker == "user":
        yield from _SENTENCE_END.split(" ".join(span))


def _same_matter(a: Claim, d: Claim, shared: frozenset) -> bool:
    """Same stemmed verb on both sides, or at least two shared words outside GENERIC."""
    if a.verb and a.verb == d.verb:
        return True
    return sum(1 for w in shared if w not in GENERIC) >= 2


class StateLedger:
    """Index user claims from dialogue units; pair an affirmation with a denial when a gated question asks."""

    def __init__(self) -> None:
        self.claims: List[Claim] = []

    def add(self, unit_id: str, text: str, date: str = "") -> int:
        """Index one unit. Returns the number of claims recorded from it."""
        before = len(self.claims)
        for sentence in _user_sentences(text):
            sentence = sentence.strip()
            if not sentence:
                continue
            aff = any(p.search(sentence) for p in AFFIRM)
            den = any(p.search(sentence) for p in DENY)
            if aff == den:  # neither, or both in one sentence: not a clean claim
                continue
            self.claims.append(Claim(unit_id=unit_id, date=date or "", polarity=1 if aff else -1, text=sentence,
                                     words=content_words(sentence),
                                     verb=_verb_of(sentence, AFFIRM if aff else DENY)))
        return len(self.claims) - before

    def add_units(self, units: Mapping[str, Tuple[str, str]]) -> int:
        """``{unit_id: (text, date)}``. Returns total claims recorded."""
        return sum(self.add(uid, text, date) for uid, (text, date) in units.items())

    def conflicts_for(self, question: str) -> List[StatePair]:
        """Affirmation/denial pairs that share >= MIN_SHARED content words with each other and with the question.
        Empty unless the question passes QUESTION_GATE. Deterministic: most shared words first, then index order."""
        if not QUESTION_GATE.match(question.strip()):
            return []
        q = content_words(question)
        if len(q) < MIN_SHARED:
            return []
        aff = [(c, q & c.words) for c in self.claims if c.polarity == 1 and len(q & c.words) >= MIN_SHARED]
        den = [(c, q & c.words) for c in self.claims if c.polarity == -1 and len(q & c.words) >= MIN_SHARED]
        pairs: List[StatePair] = []
        for a, ao in aff:
            for d, do in den:
                shared = ao & do
                if len(shared) >= MIN_SHARED and a.unit_id != d.unit_id and _same_matter(a, d, shared):
                    pairs.append(StatePair(affirm=a, deny=d, shared=tuple(sorted(shared))))
        pairs.sort(key=lambda p: -len(p.shared))  # stable: ties keep index order
        return pairs

    def conflicts_for_entity(self, terms: Sequence[str]) -> List[StatePair]:
        """Programmatic form: pairs about the given content terms, with no question gate. For callers that
        already know what they are asking about (an API consumer, a planner), not for prompt injection."""
        q = frozenset(w.lower() for w in terms)
        return self.conflicts_for("Have I " + " ".join(sorted(q))) if len(q) >= MIN_SHARED else []


def header(pairs: Sequence[StatePair], max_pairs: int = 2, max_words: int = 40) -> str:
    """The text placed before the retrieved context: both statements of each pair, dated, oldest first, nothing
    else. No instruction, no tag, no bracket. Empty string when there are no pairs."""
    if not pairs:
        return ""
    lines: List[str] = []
    seen = set()
    for p in pairs[:max_pairs]:
        for c in sorted((p.affirm, p.deny), key=lambda c: (c.date, c.unit_id)):
            key = (c.unit_id, c.text)
            if key in seen:
                continue
            seen.add(key)
            words = c.text.split()
            quote = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
            when = f"On {c.date}, " if c.date else ""
            lines.append(f'{when}user said: "{quote}"')
    return "\n".join(lines)


__all__ = ["AFFIRM", "DENY", "QUESTION_GATE", "MIN_SHARED", "GENERIC", "Claim", "StatePair", "StateLedger",
           "content_words", "header", "stem"]
