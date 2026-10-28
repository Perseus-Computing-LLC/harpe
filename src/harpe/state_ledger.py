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

Everything here is lexical and deterministic. Freeze the file by SHA-256 before any paid inference.

0.1.0a9 (after Round 8 reported null, 0.3384 vs 0.3384): claims carry ``seq`` (insertion order) so same-date or
undated claims sort by arrival instead of by unit-id string; pairs rank by length-normalised overlap so a tight
polarity flip outranks a long request sentence; ``conflicts_for_entity`` takes a date window, a threshold and a
limit; ``latest_state`` and ``timeline`` expose the same index as data. The prompt path (``header``) is unchanged.
"""

from __future__ import annotations

import re
import math
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

_VERBS = (
    r"completed|implemented|created|fixed|written|set up|developed|built|configured|tested|solved|attended|met|"
    r"signed|updated|purchased|taken"
)
_SUBJ = r"(?:i(?:'ve| have)?|we(?:'ve| have)?)"

AFFIRM = (
    re.compile(rf"\b{_SUBJ}\s+(?:{_VERBS})\b", re.IGNORECASE),
    re.compile(rf"\b(?:i|we)\s+(?:always|usually)\b", re.IGNORECASE),
    re.compile(rf"\byes,?\s+{_SUBJ}\b", re.IGNORECASE),
)
DENY = (
    re.compile(rf"\b{_SUBJ}\s+(?:never|not yet)\s+(?:{_VERBS})\b", re.IGNORECASE),
    re.compile(rf"\b(?:i|we)\s+(?:never|rarely)\b", re.IGNORECASE),
    re.compile(rf"\b(?:haven't|have not|didn't|did not|never)\s+(?:{_VERBS})\b", re.IGNORECASE),
    re.compile(r"\b(?:refused to|decided against|avoided)\b", re.IGNORECASE),
)
# The question gate. A header is only considered for questions that ask about the user's own past action or state.
QUESTION_GATE = re.compile(r"^(?:have (?:i|you)|has [a-z]+|do i|did i|how experienced am i)\b", re.IGNORECASE)
MIN_SHARED = 2  # content words the question, the claim and the denial must all share

_STOP = frozenset("the a an in on at of to for with by from is was are were and or but so that it this have has had "
                  "been be".split())
_WORD = re.compile(r"\b[a-z]{3,}\b")
_ROLE = re.compile(r"^\s*(user|assistant|system)\s*:\s*", re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=\S)")


def content_words(text: str) -> frozenset:
    return frozenset(w for w in _WORD.findall(text.lower()) if w not in _STOP)


class Claim(BaseModel):
    """One user sentence that affirms (+1) or denies (-1) a first-person action or state.

    ``seq`` is the claim's position in insertion order; ``(date, seq)`` is the ledger's time order. ``unit_id`` is
    a caller string and never sorted on ("10:2" would precede "2:1")."""

    unit_id: str
    date: str = ""
    polarity: int = Field(ge=-1, le=1)
    text: str
    words: frozenset = Field(default_factory=frozenset)
    seq: int = 0

    model_config = {"arbitrary_types_allowed": True, "frozen": True}

    @property
    def order(self) -> Tuple[str, int]:
        return (self.date, self.seq)


class StatePair(BaseModel):
    affirm: Claim
    deny: Claim
    shared: Tuple[str, ...]

    @property
    def strength(self) -> float:
        """Length-normalised overlap: |shared|^2 / sqrt(|affirm.words| * |deny.words|). A two-word flip between two
        short sentences scores higher than two words buried in a forty-word request."""
        return pair_strength(len(self.shared), len(self.affirm.words), len(self.deny.words))


def pair_strength(shared: int, affirm_words: int, deny_words: int) -> float:
    denominator = math.sqrt(max(affirm_words, 1) * max(deny_words, 1))
    return (shared * shared) / denominator


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
                                     words=content_words(sentence), seq=len(self.claims)))
        return len(self.claims) - before

    def add_units(self, units: Mapping[str, Tuple[str, str]]) -> int:
        """``{unit_id: (text, date)}``. Returns total claims recorded."""
        return sum(self.add(uid, text, date) for uid, (text, date) in units.items())

    def _pairs(self, q: frozenset, since: Optional[str], until: Optional[str], min_shared: int) -> List[StatePair]:
        """Affirmation/denial pairs whose shared content words with ``q`` and with each other number at least
        ``min_shared``. Ranked by ``pair_strength`` descending; ties keep ledger order (affirm seq, then deny seq)."""
        if len(q) < min_shared:
            return []
        claims = [c for c in self.claims if _in_window(c.date, since, until)]
        aff = [(c, q & c.words) for c in claims if c.polarity == 1 and len(q & c.words) >= min_shared]
        den = [(c, q & c.words) for c in claims if c.polarity == -1 and len(q & c.words) >= min_shared]
        pairs: List[StatePair] = []
        for a, ao in aff:
            for d, do in den:
                shared = ao & do
                if len(shared) >= min_shared and a.unit_id != d.unit_id:
                    pairs.append(StatePair(affirm=a, deny=d, shared=tuple(sorted(shared))))
        pairs.sort(key=lambda p: (-p.strength, p.affirm.seq, p.deny.seq))
        return pairs

    def conflicts_for(self, question: str) -> List[StatePair]:
        """Prompt path. Pairs that share >= MIN_SHARED content words with each other and with the question.
        Empty unless the question passes QUESTION_GATE. Deterministic: strongest pair first, then ledger order."""
        if not QUESTION_GATE.match(question.strip()):
            return []
        return self._pairs(content_words(question), None, None, MIN_SHARED)

    def conflicts_for_entity(self, terms: Sequence[str], *, since: Optional[str] = None, until: Optional[str] = None,
                             min_shared: int = MIN_SHARED, limit: Optional[int] = None) -> List[StatePair]:
        """Programmatic path: pairs about the given content terms, with no question gate. For callers that already
        know what they are asking about (an API consumer, a planner), not for prompt injection. ``since``/``until``
        are inclusive ISO dates applied to both claims; undated claims pass any window."""
        pairs = self._pairs(_terms(terms), since, until, min_shared)
        return pairs if limit is None else pairs[:limit]

    def timeline(self, terms: Sequence[str], *, since: Optional[str] = None, until: Optional[str] = None,
                 min_shared: int = 1) -> List[Claim]:
        """Every claim of either polarity that shares >= ``min_shared`` content words with ``terms``, ordered by
        ``(date, seq)`` - undated claims first, in arrival order."""
        q = _terms(terms)
        hits = [c for c in self.claims if len(q & c.words) >= min_shared and _in_window(c.date, since, until)]
        return sorted(hits, key=lambda c: c.order)

    def latest_state(self, terms: Sequence[str], *, until: Optional[str] = None, min_shared: int = 1) -> Optional[Claim]:
        """The most recent claim on the topic, either polarity, by ``(date, seq)``; None when nothing matches."""
        hits = self.timeline(terms, until=until, min_shared=min_shared)
        return hits[-1] if hits else None


def _terms(terms: Sequence[str]) -> frozenset:
    return frozenset(w for t in terms for w in content_words(t))


def _in_window(date: str, since: Optional[str], until: Optional[str]) -> bool:
    if not date:
        return True
    return (since is None or date >= since) and (until is None or date <= until)


def header(pairs: Sequence[StatePair], max_pairs: int = 2, max_words: int = 40) -> str:
    """The text placed before the retrieved context: both statements of each pair, dated, oldest first, nothing
    else. No instruction, no tag, no bracket. Empty string when there are no pairs."""
    if not pairs:
        return ""
    lines: List[str] = []
    seen = set()
    for p in pairs[:max_pairs]:
        for c in sorted((p.affirm, p.deny), key=lambda c: c.order):
            key = (c.unit_id, c.text)
            if key in seen:
                continue
            seen.add(key)
            words = c.text.split()
            quote = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
            when = f"On {c.date}, " if c.date else ""
            lines.append(f'{when}user said: "{quote}"')
    return "\n".join(lines)


__all__ = ["AFFIRM", "DENY", "QUESTION_GATE", "MIN_SHARED", "Claim", "StatePair", "StateLedger", "content_words",
           "header", "pair_strength"]
