"""Conflict surfacing: a statement in the selected context that another statement denies.

A reader handed two remembered statements, "I've written integration tests for socket events" and, later,
"I've never written integration tests for any socket events", tends to answer from one of them. Flagging the
pair lets it say that the memory disagrees, or prefer the later statement, instead of picking one at random.
Detection is lexical and deterministic: a sentence with a denial ("never", "haven't", "didn't", ...) and a
sentence without one, from different items and the same speaker side, sharing most of their content words.
"""

from __future__ import annotations

import re
from typing import Dict, List, Mapping, Optional, Tuple

from pydantic import BaseModel, Field

_DENIAL = re.compile(
    r"\b(?:never|haven't|have not|hasn't|has not|hadn't|had not|didn't|did not|don't|do not|doesn't|does not|"
    r"not yet|no experience|wasn't|weren't)\b",
    re.IGNORECASE,
)
_ROLE = re.compile(r"^\s*([A-Za-z][\w .'-]{0,40}?):\s+")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=\S)")
_WORD = re.compile(r"[a-z][a-z0-9'.-]*[a-z0-9]|[a-z]")
_NOT_SPEAKERS = frozenset({"assistant", "system", "ai", "bot", "model", "tool"})
_STOP = frozenset(
    """a about above after again against all also am an and any are as at be been before being below between both
    but by can could did do does doing done down during each even ever few for from further get got had has have
    having he her here hers herself him himself his how i i'd i'll i'm i've if in into is it it's its itself just
    let's like me might more most must my myself need no nor not now of off on once only or other our ours
    ourselves out over own really same she should so some such sure than that that's the their theirs them
    themselves then there these they this those through to too try trying tried under until up us use used using
    very want was we were what when where which while who whom why will with would yet you your yours yourself
    yourselves start started where's lot lots thing things way much many one two help""".split()
)
_SUFFIXES = ("izations", "ization", "ations", "ation", "izing", "ized", "izes", "ize", "ings", "ing", "edly", "ed",
             "es", "s")


class Conflict(BaseModel):
    """Two items' statements that disagree: one denies what the other states."""
    denied_id: str = Field(description="Item whose sentence denies (\"I've never ...\").")
    stated_id: str = Field(description="Item whose sentence states it.")
    denied: str = Field(description="The denying sentence, as it appears in its item (speaker prefix removed).")
    stated: str = Field(description="The stating sentence.")
    stated_speaker: str = Field(default="", description="Speaker of the stating sentence, when the item names one.")
    shared_terms: List[str] = Field(default_factory=list, description="Content words the two sentences share.")


def _stem(word: str) -> str:
    word = word.strip("'.-")
    if word.endswith("'s"):
        word = word[:-2]
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def _terms(text: str) -> Dict[str, str]:
    """Stemmed content words -> the first surface form."""
    out: Dict[str, str] = {}
    for word in _WORD.findall(text.lower()):
        if len(word) < 3 or word in _STOP or _DENIAL.fullmatch(word):
            continue
        out.setdefault(_stem(word), word)
    return out


def _sentences(text: str) -> List[Tuple[str, str]]:
    """(speaker, sentence) pairs; lines without a speaker prefix continue the previous speaker."""
    speaker = ""
    out: List[Tuple[str, str]] = []
    for line in text.splitlines():
        m = _ROLE.match(line)
        if m:
            speaker, line = m.group(1).strip(), line[m.end():]
        for sentence in _SENTENCE_END.split(line.strip()):
            if sentence.strip():
                out.append((speaker, sentence.strip()))
    return out


def find_conflicts(
    texts: Mapping[str, str],
    query: Optional[str] = None,
    min_shared: int = 3,
    min_overlap: float = 0.4,
    max_conflicts: int = 3,
) -> List[Conflict]:
    """Pairs of sentences in different items where one denies what the other states.

    ``texts`` maps item ids to their text, in context order. Sentences spoken by an assistant, system or tool
    are ignored. A pair counts when the sentences share at least ``min_shared`` content words, which make up at
    least ``min_overlap`` of the shorter sentence's content words. With a ``query``, both sentences must also
    use at least two of its content words (one, if it has only one), so that only conflicts on the asked
    topic are flagged. At most ``max_conflicts`` pairs, the strongest denial first.
    """
    q_terms = set(_terms(query)) if query else set()
    need_q = min(2, len(q_terms))
    found: List[Tuple[int, int, Conflict]] = []
    order = {item: i for i, item in enumerate(texts)}
    parsed = {
        item: [(sp, s, _terms(s), bool(_DENIAL.search(s))) for sp, s in _sentences(text)
               if sp.lower() not in _NOT_SPEAKERS]
        for item, text in texts.items()
    }
    for d_item, d_sentences in parsed.items():
        for _, d_text, d_terms, d_denies in d_sentences:
            if not d_denies or len(d_terms) < min_shared:
                continue
            if q_terms and len(q_terms.intersection(d_terms)) < need_q:
                continue
            best: Optional[Tuple[int, Conflict]] = None
            for s_item, s_sentences in parsed.items():
                if s_item == d_item:
                    continue
                for s_speaker, s_text, s_terms, s_denies in s_sentences:
                    if s_denies:
                        continue
                    shared = [d_terms[k] for k in d_terms if k in s_terms]
                    if len(shared) < min_shared or len(shared) < min_overlap * min(len(d_terms), len(s_terms)):
                        continue
                    if q_terms and len(q_terms.intersection(s_terms)) < need_q:
                        continue
                    if best is None or len(shared) > best[0]:
                        best = (len(shared), Conflict(denied_id=d_item, stated_id=s_item, denied=d_text,
                                                      stated=s_text, stated_speaker=s_speaker, shared_terms=shared))
            if best is not None:
                found.append((-best[0], order[d_item], best[1]))
    found.sort(key=lambda x: (x[0], x[1]))
    seen = set()
    out: List[Conflict] = []
    for _, _, conflict in found:
        key = (conflict.denied_id, conflict.stated_id)
        if key in seen:
            continue
        seen.add(key)
        out.append(conflict)
        if len(out) == max_conflicts:
            break
    return out


def conflict_note(conflict: Conflict, stated_date: Optional[str] = None, max_words: int = 30) -> str:
    """``[possible conflict: on <date>, <speaker> said "<statement>"]``, to follow the denying sentence."""
    words = conflict.stated.split()
    quote = " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")
    who = conflict.stated_speaker or "it was"
    when = f"on {stated_date}, " if stated_date else ""
    said = "said" if conflict.stated_speaker else "said that"
    return f'[possible conflict: {when}{who} {said} "{quote}"]'


def annotate_conflicts(
    texts: Mapping[str, str], conflicts: List[Conflict], dates: Optional[Mapping[str, str]] = None
) -> Dict[str, str]:
    """``texts`` with each conflict's note placed after its denying sentence (first occurrence)."""
    out = dict(texts)
    for conflict in conflicts:
        text = out.get(conflict.denied_id)
        if text is None or conflict.denied not in text:
            continue
        note = conflict_note(conflict, (dates or {}).get(conflict.stated_id))
        if note in text:
            continue
        out[conflict.denied_id] = text.replace(conflict.denied, f"{conflict.denied} {note}", 1)
    return out
