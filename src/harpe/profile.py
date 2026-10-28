"""Profile ledger: the user's own statements about profile attributes, resolved by mutability.

Asked "How did the user's residence change?" over forty retrieved dialogue rounds, a reader tends to quote
whichever mention it meets first, or hedges across all of them. The ledger collects, per attribute, the
sentences in which the user states something about that attribute, each with its date, and resolves them by the
attribute's mutability:

* mutable attributes (where the user lives, their career, relationship, workload, social life, health,
  children): the latest statement is current and earlier ones are superseded;
* immutable attributes (birthplace, birth date, gender, education, parents, siblings): the first statement is the
  established record. A later statement that names a different value is flagged as conflicting; it does not
  replace the record.

Detection is lexical and deterministic: first-person sentence patterns over the user's own turns, no model calls.
The ledger only holds what was added to it, so a reader sees nothing the user had not said by then.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------------------------------------------
# Text helpers
# --------------------------------------------------------------------------------------------------------------
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'\u201c\u2018(])|\s*[;\n]+\s*|\s+[\u2014\u2013]\s+|\s*\u2014\s*|"
                             r"\s+-{1,2}\s+")
_CONTINUATION = re.compile(r"^(?:he|she|they|his|her|their|it)\b", re.I)
_FIRST_PERSON = re.compile(r"\b(?:i|i'm|i\u2019m|im|i've|i\u2019ve|i'd|i'll|me|my|mine|myself|we|we're|we've|our|us)\b",
                           re.I)
# A sentence that talks about a state the user has left ("I used to live in Darwin") is never the current one.
_PAST = re.compile(r"\b(?:(?<!than\si\s)(?<!than\swe\s)(?<!than\s)used\s+to|previously|formerly|back\s+when|no\s+longer|"
                   r"anymore|any\s+more|"
                   r"before\s+(?:i|we)\s+(?:moved|relocated|left|switched|started|changed)|"
                   r"(?:my|our)\s+(?:old|former|previous)\b)", re.I)
# A statement whose subject is someone else ("my dad majored in ...", "she moved to ...") is not about the user.
_THIRD_PARTY = re.compile(
    r"\b(?:he|she|they|his|her|their|(?:my|our)\s+(?:\w+\s+)?(?:dad|father|mom|mother|mum|parents?|brother|sister|"
    r"siblings?|friends?|partner|wife|husband|spouse|girlfriend|boyfriend|sons?|daughters?|kids?|child|children|"
    r"colleagues?|coworkers?|co-workers?|boss|manager|neighbou?rs?|cousins?|aunt|uncle|grand\w+|roommate|flatmate|"
    r"fianc\w+|ex))\b", re.I)
# A kinship noun owned by someone else ("her father", "Claire's dad") is theirs, not the user's. Only consulted for a
# bare noun; "my dad" is always the user's. Sentence-opening contractions ("There's") are not names.
_OTHERS_POSSESSIVE = re.compile(r"(?:\b(?:her|his|their|its)|\b(?!(?:There|It|That|This|He|She|What|Here|Let|Who|Where|"
                                r"How|Everyone|Someone|Everything|Nothing|Something)['\u2019])[A-Z][a-z]+['\u2019]s)\s+(?:\w+\s+)?$")
# "the father" is someone else's only when the sentence brings in a third person ("...when Claire is with him");
# "my birth records list the mother as unknown" is about the user.
_THE_KIN = re.compile(r"\bthe\s+(?:\w+\s+)?$")
_OTHER_PERSON = re.compile(r"\b(?:her|his|their|him|them|she|he|they)\b|(?<!^)(?<![.!?]\s)\b[A-Z][a-z]+\b")
# "a friend assumed I didn't have siblings, which isn't true": a reported or denied negative is not the user's own "no".
_REPORTED_NEGATIVE = re.compile(r"\b(?:assumed|thought|figured|believed|guessed|claimed|said|says|insisted)\s+(?:that\s+)?I\b|"
                                r"\b(?:isn['\u2019]t|is\s+not|wasn['\u2019]t|was\s+not)\s+true\b|"
                                r"\bwhich\s+(?:isn['\u2019]t|is\s+not)\s+(?:true|the\s+case)\b", re.I)
_SELF = re.compile(r"\b(?:i|i'm|i\u2019m|im|i've|i\u2019ve|i'd|we|we're|we've|me|myself)\b", re.I)


def _about_someone_else(sentence: str, start: int) -> bool:
    """True when the nearest subject before a match is a third party and the match itself is not about "I"."""
    if _SELF.match(sentence, start):
        return False
    window = sentence[max(0, start - 70):start]
    if re.search(r"\b(?:who|whom|whose|which)\s+(?:\w+\s+)?$", window, re.I):
        return True  # "a colleague ... who went to the University of Toronto"
    last = None
    for m in _THIRD_PARTY.finditer(window):
        last = m
    return last is not None and not _SELF.search(window[last.end():])


_CONDITIONAL_QUESTION = re.compile(
    r"\b(?:under|in|on)\s+what\s+(?:conditions?|circumstances?|situations?|contexts?|occasions?)\b|"
    r"\bwhat\s+(?:conditions?|circumstances?|situations?)\b|\bwhen\s+does\s+(?:the\s+)?user\s+(?:prefer|like|choose|want)\b",
    re.I)
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\u2019&.-]*")

_CAP_STOP = frozenset("""
i i'm i've i'd i'll im oh hey hi hello also quick just so but and or by the a an it its it's my me we our also
actually anyway btw well yes no okay ok thanks thank sure maybe still now then here there this that these those
what when where which who why how can could would should will do does did is are was were be been have has had
for from with without about after before during since until while if though although because as at in on of to
mom mum mother dad father mama papa brother sister sibling siblings family friend friends monday tuesday wednesday
thursday friday saturday sunday january february march april may june july august september october november
december one two three four five six seven eight nine ten first second third update note fyi lately recently
he she they them him his her hers their theirs you your yours us hr ceo
""".split())
_NUMBER_WORDS = {"zero": "0", "no": "0", "none": "0", "one": "1", "a": "1", "an": "1", "single": "1", "two": "2",
                 "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
                 "ten": "10", "twin": "2", "twins": "2"}
_MONTHS = ("january|february|march|april|may|june|july|august|september|october|november|december|"
           "jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec")


def split_sentences(text: str) -> List[str]:
    """Sentences of one turn, with a sentence that starts with a pronoun joined to the one before it."""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text or "") if p and p.strip()]
    out: List[str] = []
    for part in parts:
        if out and _CONTINUATION.match(part) and len(out[-1].split()) < 40:
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def _proper_nouns(text: str) -> FrozenSet[str]:
    """Capitalised words and acronyms that are not sentence-initial filler: the names a statement commits to."""
    words = _WORD.findall(text)
    found = set()
    for i, word in enumerate(words):
        bare = word.strip(".'\u2019-").rstrip("s") if word.endswith("'s") else word.strip(".'\u2019-")
        if not bare or not bare[0].isupper():
            continue
        if bare.lower().replace("\u2019", "'") in _CAP_STOP:
            continue
        if i == 0 and not bare.isupper():
            continue
        found.add(bare.lower())
    return frozenset(found)


def _years(text: str) -> FrozenSet[str]:
    return frozenset(re.findall(r"\b(1[89]\d{2}|20\d{2})\b", text))


def _content_after(text: str, start: int, n: int = 6) -> FrozenSet[str]:
    words = [w.lower().strip(".,'\u2019") for w in _WORD.findall(text[start:])]
    keep = [w for w in words if len(w) > 3 and w not in _CAP_STOP and not w.isdigit()]
    return frozenset(keep[:n])


# --------------------------------------------------------------------------------------------------------------
# Value extractors for immutable attributes. A value is a frozenset of tokens; an empty set means "no value read".
# --------------------------------------------------------------------------------------------------------------
_GENDER_MALE = re.compile(r"\b(?:i'?m|i\s+am|i\u2019m|identify\s+as|present\s+as|as)\s+(?:a\s+|an\s+)?(?:guy|man|male|"
                          r"gentleman|dude|boy|husband|father|dad)\b|\bhe/him\b", re.I)
_GENDER_FEMALE = re.compile(r"\b(?:i'?m|i\s+am|i\u2019m|identify\s+as|present\s+as|as)\s+(?:a\s+|an\s+)?(?:woman|"
                            r"female|girl|lady|wife|mother|mom|mum)\b|\bshe/her\b", re.I)
_GENDER_OTHER = re.compile(r"\b(?:non-?binary|genderqueer|they/them)\b", re.I)
_DEGREE_LEVELS = (("doctorate", r"\b(?:ph\.?\s?d|doctorate|doctoral|doctor\s+of)\b"),
                  ("master", r"\b(?:master'?s?|m\.?sc|m\.?s\.|mba|m\.?a\.|master\s+of)\b"),
                  ("bachelor", r"\b(?:bachelor'?s?|b\.?sc|b\.?s\.|b\.?a\.|undergrad(?:uate)?\s+degree|bachelor\s+of)\b"),
                  ("associate", r"\bassociate'?s?\s+degree\b"),
                  ("high-school", r"\bhigh\s+school\s+(?:diploma|graduate)\b"))
# "I have no X" in the user's own words. Straight and curly apostrophes both occur in transcripts ("don't", "don\u2019t");
# a qualifier may sit between the negation and the noun ("no known biological mother", "don't really have siblings").
_NEG_HAVE = r"(?:don['\u2019]?t|do\s+not|didn['\u2019]?t|did\s+not|never|haven['\u2019]?t|have\s+not)\s+(?:really\s+|actually\s+)?(?:have|had|knew|know|known)"
_QUAL = r"(?:(?:living|recorded|known|biological|birth|legal|surviving|real|adoptive|present|named|listed)\s+(?:or\s+)?)*"
_HAS_NONE = {
    "siblings": re.compile(r"\b(?:only\s+child|no\s+(?:brothers|sisters|siblings)|" + _NEG_HAVE +
                           r"\s+(?:any\s+)?" + _QUAL + r"(?:siblings|brothers|sisters)|without\s+(?:any\s+)?siblings|"
                           r"not\s+having\s+(?:any\s+)?siblings)\b", re.I),
    "mother": re.compile(r"\b(?:" + _NEG_HAVE + r"\s+(?:a\s+|my\s+|any\s+)?" + _QUAL + r"(?:mother|mom|mum|maternal\s+(?:parent|relationship))|"
                         r"without\s+(?:a\s+|my\s+|any\s+)?" + _QUAL + r"(?:mother|mom|mum|maternal\s+(?:parent|relationship))|"
                         r"not\s+knowing\s+my\s+(?:mother|mom|mum)|lost\s+my\s+(?:mother|mom|mum)|"
                         r"(?:mother|mom|mum|she)\s+(?:passed(?:\s+away)?|died|is\s+deceased|has\s+passed)\b|"
                         r"no\s+" + _QUAL + r"(?:mother|mom|mum|maternal\s+(?:parent|relationship|records?))\b|"
                         r"no\s+record\s+of\s+(?:a\s+|my\s+)?" + _QUAL + r"(?:mother|mom|mum)|"
                         r"raised\s+(?:solely|only|entirely)\s+by\s+(?:my\s+)?(?:father|dad)\b|"
                         r"(?:mother|mom|mum)\s+(?:is\s+|was\s+)?(?:listed|recorded|registered)\s+as\s+['\u2018\u2019\"]?unknown|"
                         r"(?:lists?|listed|records?|recorded)\s+(?:the|my)\s+(?:mother|mom|mum)\s+as\s+['\u2018\u2019\"]?unknown|"
                         r"both\s+(?:birth\s+|biological\s+)?parents\s+(?:were|are|being)\s+absent|orphan)", re.I),
    "father": re.compile(r"\b(?:" + _NEG_HAVE + r"\s+(?:a\s+|my\s+|any\s+)?" + _QUAL + r"(?:father|dad|paternal\s+(?:parent|relationship))|"
                         r"without\s+(?:a\s+|my\s+|any\s+)?" + _QUAL + r"(?:father|dad|paternal\s+(?:parent|relationship))|"
                         r"not\s+knowing\s+my\s+(?:father|dad)|lost\s+my\s+(?:father|dad)|"
                         r"(?:father|dad|he)\s+(?:passed(?:\s+away)?|died|is\s+deceased|has\s+passed)\b|"
                         r"no\s+" + _QUAL + r"(?:father|dad|paternal\s+(?:parent|relationship|records?))\b|"
                         r"no\s+record\s+of\s+(?:a\s+|my\s+)?" + _QUAL + r"(?:father|dad)|"
                         r"raised\s+(?:solely|only|entirely)\s+by\s+(?:my\s+)?(?:mother|mom|mum)\b|"
                         r"(?:father|dad)\s+(?:is\s+|was\s+)?(?:listed|recorded|registered)\s+as\s+['\u2018\u2019\"]?unknown|"
                         r"(?:lists?|listed|records?|recorded)\s+(?:the|my)\s+(?:father|dad)\s+as\s+['\u2018\u2019\"]?unknown|"
                         r"both\s+(?:birth\s+|biological\s+)?parents\s+(?:were|are|being)\s+absent|orphan)", re.I),
}
_SIBLING_COUNT = re.compile(r"\b(zero|no|one|two|three|four|five|six|\d|a|an|single)\s+(?:older\s+|younger\s+|little\s+|"
                            r"big\s+)?(?:siblings?|brothers?|sisters?)\b|\b(twins?)\b", re.I)
_SIBLING_KIND = re.compile(r"\b(brother|sister)s?\b", re.I)


def _gender_value(text: str, start: int) -> FrozenSet[str]:
    if _GENDER_OTHER.search(text):
        return frozenset({"nonbinary"})
    male, female = bool(_GENDER_MALE.search(text)), bool(_GENDER_FEMALE.search(text))
    if male != female:
        return frozenset({"male" if male else "female"})
    return frozenset()


def _degree_value(text: str, start: int) -> FrozenSet[str]:
    for level, pattern in _DEGREE_LEVELS:
        if re.search(pattern, text, re.I):
            return frozenset({level})
    return frozenset()


def _family_value(kind: str) -> Callable[[str, int], FrozenSet[str]]:
    def read(text: str, start: int) -> FrozenSet[str]:
        values = set()
        reported = bool(_REPORTED_NEGATIVE.search(text))
        values.add("has:no" if _HAS_NONE[kind].search(text) and not reported else "has:yes")
        if kind == "siblings" and not reported:
            for word, twin in _SIBLING_COUNT.findall(text):
                raw = str(word or twin).lower()
                number = _NUMBER_WORDS.get(raw, raw)
                if number.isdigit():
                    values.add(f"count:{number}")
            kinds = {k.lower() for k in _SIBLING_KIND.findall(text)}
            if len(kinds) == 1:
                values.add(f"kind:{kinds.pop()}")
        values |= {f"year:{y}" for y in _years(text)}
        values |= {f"name:{n}" for n in _proper_nouns(text)}
        return frozenset(values)
    return read


_MONTH_NUMBER = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10,
                 "nov": 11, "dec": 12}
_ISO_DATE = re.compile(r"\b((?:19|20)\d{2})[-/.](\d{1,2})[-/.](\d{1,2})\b")
_MONTH_DAY_YEAR = re.compile(r"\b(" + _MONTHS + r")\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+((?:19|20)\d{2})\b", re.I)
_DAY_MONTH_YEAR = re.compile(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(" + _MONTHS + r")\.?,?\s+((?:19|20)\d{2})\b", re.I)


def _full_dates(text: str) -> FrozenSet[str]:
    """Calendar dates written out in full ("1980-03-22", "March 22, 1980", "22 March 1980") as date:YYYY-MM-DD."""
    triples = list(_ISO_DATE.findall(text))
    triples += [(y, str(_MONTH_NUMBER[mon.lower()[:3]]), d) for mon, d, y in _MONTH_DAY_YEAR.findall(text)]
    triples += [(y, str(_MONTH_NUMBER[mon.lower()[:3]]), d) for d, mon, y in _DAY_MONTH_YEAR.findall(text)]
    return frozenset(f"date:{y}-{int(m):02d}-{int(d):02d}" for y, m, d in triples
                     if 1 <= int(m) <= 12 and 1 <= int(d) <= 31)


def _birthdate_value(text: str, start: int) -> FrozenSet[str]:
    """Years, plus the full date when one is written out: two dates in the same year are still different dates."""
    return frozenset(_years(text)) | _full_dates(text)


def _names_value(text: str, start: int) -> FrozenSet[str]:
    after = _proper_nouns(text[max(0, start - 1):])
    if after:
        return after
    # "... list Cleveland, Ohio as my birthplace": the value comes before the trigger
    before = text[max(0, start - 80):start]
    return _proper_nouns(before) if re.search(r"\bas\s+(?:my\s+)?$", before, re.I) else frozenset()


def _major_value(text: str, start: int) -> FrozenSet[str]:
    return _content_after(text, start, 6)


_INSTITUTION = re.compile(
    r"(?-i:(?:[A-Z][\w&.'\u2019-]*\s+(?:(?:of|at|for|and|&|de|la)\s+)?){0,5}(?:University|College|Institute|Polytechnic|"
    r"Academy|School)(?:\s+(?:of|at|in)\s+[A-Z][\w.'-]*(?:[\s,]+[A-Z][\w.'-]*){0,3})?|\b[A-Z]{2,6}\b(?:\s+[A-Z][a-z]+)?)")


def _institution_value(text: str, start: int) -> FrozenSet[str]:
    names = set()
    for m in _INSTITUTION.finditer(text):
        tokens = {w.lower().strip(".,'\u2019") for w in m.group(0).split()}
        tokens -= {"university", "college", "institute", "school", "academy", "polytechnic", "of", "at", "in", "the",
                   "and", "&", "for", "i", "i'm", "my", "bs", "ba", "ms", "ma", "mba", "phd", "b.s", "m.s", "usa", "us"}
        if tokens:
            names |= tokens
    return frozenset(names)


_PROPER_NAME = (r"(?-i:[A-Z][a-z]+(?:-[A-Za-z]+)?(?:\s+(?:(?:de|van|von|da|del|der|la|le|al|bin|ibn)\s+)?"
                r"[A-Z][a-z]+(?:-[A-Za-z]+)?)+)")
_NAME_TRIGGER = (r"(?:\b(?:i'?m|i\u2019m|i\s+am|my\s+name\s+is|my\s+name'?s|my\s+name\u2019s|call\s+me)|"
                 r"(?:^|\b(?:hey|hi|hello)\b[\s,!\u2014\u2013-]*)(?:it'?s|it\u2019s|this\s+is))\s+")
_NAME_CAPTURE = re.compile(_NAME_TRIGGER + "(" + _PROPER_NAME + ")", re.I)


def _person_name_value(text: str, start: int) -> FrozenSet[str]:
    match = _NAME_CAPTURE.search(text, max(0, start))
    if not match:
        return frozenset()
    return frozenset(w.lower() for w in match.group(1).split()
                     if w.lower() not in {"de", "van", "von", "da", "del", "der", "la", "le", "al", "bin", "ibn"})


_CORRECTION = re.compile(r"\b(?:actually|exclusively|only|never|not|instead|rather|but|however|officially|legally|"
                         r"records?|documents?|certificate|transcripts?|paperwork|passport|identification|really|"
                         r"in\s+fact|truth|correction)\b|n't\b|n\u2019t\b", re.I)


def _values_conflict(attribute: str, first: FrozenSet[str], later: FrozenSet[str], text: str = "") -> Optional[bool]:
    """True: the later statement names a different value. False: same value. None: cannot tell."""
    if not first or not later:
        return None
    if attribute == "name":
        return not (first <= later or later <= first)
    if attribute in ("siblings", "mother", "father"):
        for prefix in ("has:", "count:", "kind:", "year:"):
            a = {v for v in first if v.startswith(prefix)}
            b = {v for v in later if v.startswith(prefix)}
            if a and b and not (a & b):
                return True
        a_names = {v for v in first if v.startswith("name:")}
        b_names = {v for v in later if v.startswith("name:")}
        if a_names & b_names and b_names - a_names and a_names - b_names:
            return True  # same surname, another given name: "Robert Andrews" after "Michael Andrews"
        a_count = {v for v in first if v.startswith("count:")}
        if "has:no" in later and (a_count - {"count:0"}):
            return True
        if "has:no" in first and any(v.startswith("count:") and v != "count:0" for v in later):
            return True
        return False if ({v for v in first if v.startswith("has:")} & {v for v in later if v.startswith("has:")}) else None
    if attribute == "birthdate":
        # Compare full dates when both statements give one ("1980-03-22" against "1980-08-14" differ), else years.
        a_dates = {v for v in first if v.startswith("date:")}
        b_dates = {v for v in later if v.startswith("date:")}
        if a_dates and b_dates:
            return not (a_dates & b_dates)
        a_years = first - a_dates
        b_years = later - b_dates
        return (not (a_years & b_years)) if a_years and b_years else None
    if first & later:
        # Shares a token with the record but adds new ones under a correction ("but my records list Sheffield",
        # "I studied civil engineering exclusively"): a different value, not a restatement.
        return bool(later - first) and bool(_CORRECTION.search(text))
    return True


# --------------------------------------------------------------------------------------------------------------
# The attribute schema
# --------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AttributeSpec:
    """One profile attribute: how a user states it, how a question asks about it, and whether it changes."""

    name: str
    label: str
    mutable: bool
    statements: Tuple[str, ...]
    question: Tuple[str, ...]
    value: Optional[Callable[[str, int], FrozenSet[str]]] = None
    first_person: bool = True  # require a first-person word in the sentence
    requires_value: bool = False  # a sentence counts only when the value extractor reads a value from it


_I = r"(?:i|we)"
_IM = r"(?:i'?m|i\u2019m|i\s+am|we'?re|we\s+are)"
_IVE = r"(?:i'?ve|i\u2019ve|i\s+have|we'?ve|we\s+have)"
_ADV = (r"(?:(?:now|just|recently|finally|currently|still|officially|actually|really|also|happily|newly|always|"
        r"so|super|very|pretty|quite|mostly|basically|honestly|definitely|way|much|even|kind\s+of|a\s+bit|a\s+lot)\s+){0,3}")

DEFAULT_ATTRIBUTES: Tuple[AttributeSpec, ...] = (
    AttributeSpec(
        "name", "The user's name", False,
        statements=(_NAME_TRIGGER + _PROPER_NAME,),
        question=(r"\b(?:user'?s|user\u2019s|person'?s|their)\s+(?:full\s+|first\s+|legal\s+)?name\b|"
                  r"\bname\s+of\s+the\s+(?:user|person)\b|\bwhat\s+is\s+the\s+user\s+called\b",),
        value=_person_name_value,
        first_person=False,
        requires_value=True,
    ),
    AttributeSpec(
        "residence", "Where the user lives", True,
        statements=(
            r"\b(?:moved|relocated|relocating|relocation)\b",
            r"\bmoving\s+(?:back\s+|over\s+|out\s+|down\s+|up\s+)?(?:to|into|out|across|abroad|here|there|cities)\b",
            r"\b(?:live|lives|living|reside|residing|based|located|settled)\s+(?:in|at|near|outside|just\s+outside|"
            r"around|out\s+of)\b",
            r"\bmy\s+(?:residence|address|hometown\s+now)\b",
            r"\bmy\s+(?:new\s+)?(?:home|place|apartment|flat|house|city|neighbou?rhood)\s+(?:in|here\s+in)\b",
            r"\b(?:new|current)\s+(?:city|place|apartment|home|neighbou?rhood|address|flat)\b",
        ),
        question=(r"\b(?:residence|reside[sd]?|residing|lives?|living|move[sd]?|moving|relocat\w*|city|location|"
                  r"address|where\s+(?:does|did|is)\s+(?:the\s+)?user\s+(?:live|stay|based))\b",),
    ),
    AttributeSpec(
        "career", "The user's career (employer, job title, industry, income)", True,
        statements=(
            rf"\b{_I}\s+{_ADV}work(?:ed)?\s+(?:as|at|for|in)\b",
            rf"\b{_IM}\s+{_ADV}(?:employed|working)\s+(?:as|at|for|in|full[- ]time|part[- ]time|with)\b",
            rf"\b{_IM}\s+{_ADV}(?:an?|the)\s+(?:\w+\s+){{0,4}}(?:intern|manager|director|engineer|analyst|consultant|"
            r"designer|developer|specialist|coordinator|assistant|associate|officer|lead|senior|junior|executive|"
            r"teacher|nurse|accountant|lawyer|clerk|technician|administrator|supervisor|representative)\b",
            r"\b(?:started|starting|landed|accepted|took|got|began|start|joined|joining)\s+(?:a\s+|an\s+|my\s+|the\s+)?"
            r"(?:new\s+)?(?:\w+\s+){0,2}(?:job|role|position|internship|gig|post|company|firm|startup)\b",
            r"\b(?:new|current)\s+(?:job|role|position|internship|employer|company|title|gig)\b",
            r"\b(?:promoted|promotion|laid\s+off|got\s+fired|was\s+fired|quit\s+my\s+job|resigned|unemployed|"
            r"between\s+jobs|lost\s+(?:my|the)\s+(?:\w+\s+){0,2}job|retired|job\s+hunting|looking\s+for\s+(?:a\s+)?(?:job|work))\b",
            r"\bmy\s+(?:job|role|position|title|employer|salary|income|pay)\s+(?:is|was|as|at|now|has)\b",
            r"\b(?:i\s+)?(?:earn|make|making|earning)\s+(?:about\s+|around\s+|roughly\s+|between\s+)?\$?\d",
            r"\b(?:income|salary|pay|savings)\b[^.]{0,40}\d",
            r"\b(?:industry|sector)\b",
            r"\b(?:employment|career|job)\s+status\b",
            r"\b(?:switched|switching|changed|changing)\s+(?:jobs|careers|companies|employers|roles|fields|industries)\b",
            r"\b(?:joined|joining|started\s+at|now\s+at|working\s+at|work\s+at)\s+(?-i:[A-Z])",
        ),
        question=(r"\b(?:career|job|title|company|employer|industry|income|salary|employment|employed|occupation|"
                  r"profession|position|intern\w*|promot\w*|savings|finances|financial)\b", r"\bwork\s+status\b"),
    ),
    AttributeSpec(
        "relationship", "The user's relationship / marital status", True,
        statements=(
            rf"\b{_IM}\s+{_ADV}(?:single|married|divorced|separated|widowed|engaged|dating|seeing\s+someone|"
            r"in\s+a\s+relationship|in\s+a\s+committed)\b",
            r"\b(?:got|getting|get)\s+(?:married|engaged|divorced|hitched)\b",
            r"\b(?:broke\s+up|split\s+up|ended\s+things|filed\s+for\s+divorce|tied\s+the\s+knot|proposed|"
            r"our\s+wedding|my\s+wedding)\b",
            rf"\b{_IVE}\s+{_ADV}(?:been\s+)?(?:dating|seeing)\b",
            r"\b(?:started|start|starting|began)\s+(?:dating|seeing|a\s+relationship)\b",
            r"\b(?:became|become|becoming|now|newly|officially)\s+(?:single|divorced|married|engaged|separated|widowed)\b",
            r"\b(?:relationship|marital)\s+status\b",
            r"\bmy\s+(?:new\s+)?(?:wife|husband|spouse|partner|girlfriend|boyfriend|fianc[e\u00e9]e?|ex-?wife|"
            r"ex-?husband|ex)\b",
            r"\bmarital\s+status\b",
        ),
        question=(r"\b(?:marital|married|marriage|relationship|partner|spouse|dating|wife|husband|engaged|divorc\w*|"
                  r"single)\b",),
    ),
    AttributeSpec(
        "work_status", "The user's current work situation (busy, free, on a business trip, normal)", True,
        statements=(
            r"\bbusiness\s+trip\b", r"\b(?:travel(?:l)?ing|away|abroad)\s+for\s+work\b", r"\bwork\s+trip\b",
            rf"\b(?:{_IM}|{_IVE}\s+been|i\s+feel)\s+(?:(?:now|currently|really|so|pretty|super|quite|extremely|"
            r"very|totally|incredibly|kind\s+of|a\s+bit)\s+){0,2}(?:busy|swamped|slammed|overloaded|buried|"
            r"free|available|on\s+leave|off\s+work|on\s+sabbatical|on\s+vacation|on\s+holiday)\b",
            r"\bfree\s+from\s+work\b",
            r"\b(?:work(?:load)?|schedule)\s+(?:is|has\s+been|was|feels)\s+(?:(?:back\s+to|pretty|really|so|very)\s+)?"
            r"(?:normal|busy|light|heavy|manageable|crazy|hectic|intense|calm|lighter|heavier|quiet)\b",
            r"\bback\s+to\s+(?:my\s+)?(?:normal|regular|usual)\s+(?:work|routine|schedule|workload)\b",
            r"\b(?:normal|regular|usual)\s+(?:work\s+)?(?:schedule|routine|workload|hours|week)\b",
            r"\bsabbatical\b",
            r"\bwork\s+status\b",
            r"\b(?:workload|work\s+load|work)\s+(?:has\s+|have\s+)?(?:\w+\s+)?(?:ramped|picked\s+up|increased|exploded|"
            r"doubled|piled\s+up|lightened|eased|calmed|slowed|settled|quieted)\b",
            rf"\b{_IM}\s+{_ADV}(?:busy|swamped|slammed|overloaded|buried|free|available|traveling|travelling)\b",
            r"\b(?:busier|less\s+busy|freer|lighter\s+workload|heavier\s+workload)\b",
            r"\b(?:work|workload|job)\s+(?:has\s+|have\s+)?(?:gotten|got|become|been|is|feels)\s+(?:\w+\s+){0,2}"
            r"(?:busier|busy|heavier|lighter|crazier|calmer|intense|hectic|quieter|slower|manageable)\b",
        ),
        question=(r"\bwork\s+status\b", r"\b(?:business\s+trip|workload|busy|availability)\b"),
    ),
    AttributeSpec(
        "social", "The user's social life", True,
        statements=(
            r"\b(?:joining|joined|join)\s+(?:(?:more|some|a\s+few|new)\s+)?(?:social\s+)?(?:activities|clubs?|"
            r"meetups?|groups?|events)\b",
            r"\b(?:keeping|keep|kept)\s+to\s+(?:myself|ourselves)\b",
            r"\b(?:staying|stay|stayed|keeping|keep|kept)\s+in\s+(?:contact|touch)\b",
            r"\b(?:seeing|see|meeting|meet|met)\s+(?:(?:more|new|other)\s+)?(?:people|friends)\b",
            r"\b(?:spending|spend|spent)\s+(?:less|more)\s+time\s+(?:socializing|socialising|with\s+(?:people|friends)|"
            r"out|alone|on\s+my\s+own)\b",
            r"\b(?:socializing|socialising|socially)\b",
            r"\b(?:withdrawn|isolating|isolated|reclusive|homebody|hermit|introverted)\b",
            r"\bsocial\s+(?:life|circle|activities|events|calendar|energy|status)\b",
            r"\b(?:going|go|went)\s+out\s+(?:more|less)\b",
        ),
        question=(r"\bsocial\w*\b",),
    ),
    AttributeSpec(
        "health", "The user's health (physical and mental)", True,
        statements=(
            rf"\b(?:{_IVE}\s+been|{_IM}|i\s+feel|i\s+felt|{_IVE}\s+felt|feeling|i\s+keep\s+feeling)\s+(?:\w+\s+){{0,3}}?"
            r"(?:fatigued|tired|exhausted|drained|stressed|anxious|healthy|fit|energetic|unwell|sick|ill|distressed|"
            r"depressed|burned\s+out|burnt\s+out|run\s+down|overwhelmed|worn\s+out|emotionally\s+\w+|"
            r"physically\s+\w+|mentally\s+\w+)\b",
            r"\b(?:physically|emotionally|mentally)\s+(?:\w+\s+){0,3}?(?:fatigued|tired|exhausted|drained|stressed|"
            r"healthy|fit|energetic|unwell|distressed|positive|stable|well|strong|good|great|okay|fine|steady|calm|"
            r"balanced|down|low|solid)\b",
            r"\bmy\s+(?:health|energy|mood|mental\s+health|physical\s+health|stress\s+levels?|sleep)\b",
            r"\bhealth\s+status\b",
            r"\b(?:often|frequently|constantly|always)\s+(?:fatigued|tired|stressed|exhausted)\b",
        ),
        question=(r"\bhealth\b", r"\b(?:fatigue\w*|stress\w*|mental|physical|emotional\w*|wellbeing|well-being)\b"),
    ),
    AttributeSpec(
        "children", "The user's children", True,
        statements=(
            r"\bmy\s+(?:\w+\s+)?(?:kids?|children|sons?|daughters?|baby|newborn|toddler|child|little\s+ones)\b",
            r"\b(?:had|having|welcomed|expecting|adopted|have|got)\s+(?:(?:a|an|our|my|another|two|three|four|\d)\s+)?"
            r"(?:new\s+|little\s+|young\s+)?(?:bab(?:y|ies)|child(?:ren)?|sons?|daughters?|kids?|twins)\b",
            r"\b(?:children|kids|parenting)\s+status\b",
            r"\b(?:parent|mom|mother|mum|dad|father)\s+(?:of|to)\s+(?:\w+\s+){0,2}(?:children|kids|sons?|daughters?|"
            r"bab(?:y|ies)|twins|toddlers?|newborn)\b",
            r"\b(?:became|become|becoming|new)\s+(?:a\s+)?(?:parent|mom|mother|mum|dad|father)\b",
            r"\b(?:no|don'?t\s+have|do\s+not\s+have|never\s+had)\s+(?:any\s+)?(?:kids|children)\b",
            r"\b(?:son|daughter|baby|kid|child)\b[^.]{0,60}\bborn\b",
        ),
        question=(r"\b(?:child|children|kids?|sons?|daughters?|baby|babies|parent(?:al|hood|ing)?)\b",),
    ),
    AttributeSpec(
        "birthplace", "Where the user was born / grew up", False,
        statements=(
            r"\b(?:i\s+was\s+|i\s+am\s+|i'?m\s+|i\u2019m\s+)?born\s+(?:and\s+raised\s+)?in\s+(?!(?:19|20)\d\d|" + _MONTHS + r")",
            r"\bgrew\s+up\s+(?:in|around|near|on)\b", r"\braised\s+in\b",
            rf"\b{_IM}\s+(?:originally\s+)?from\b", r"\bmy\s+(?:hometown|home\s+town|birthplace)\b",
            r"\b\w+\s+upbringing\b", r"\bnative\s+of\b", r"\bwhere\s+i\s+(?:was\s+born|grew\s+up)\b",
        ),
        question=(r"\b(?:birthplace|born\s+in|where\s+was\s+(?:the\s+)?user(?:'s)?\s+(?:\w+\s+)?born|where\s+.*\bborn\b|"
                  r"hometown|grow\s+up|grew\s+up|place\s+of\s+birth|where\s+(?:is|was|does|did)\s+(?:the\s+)?user\s+"
                  r"(?:originally\s+)?(?:come\s+)?from)\b",),
        value=_names_value,
        requires_value=True,
    ),
    AttributeSpec(
        "birthdate", "The user's birth date", False,
        statements=(
            # A first-person subject, or "Born ..." opening the sentence: a bare "(born 2014-02-04)" after someone
            # else's name ("my kids Maya (born ...)", "dating Helen (born ...)") is their birth date, not the user's.
            r"(?:\b(?:i\s+was\s+|i'?m\s+|i\u2019m\s+|i\s+am\s+)|^\W*)born\s+(?:on\s+|in\s+)?(?:(?:19|20)\d\d|\d{1,2}(?:st|nd|rd|th)?\s+"
            r"(?:of\s+)?(?:" + _MONTHS + r")|(?:" + _MONTHS + r")\s+\d)",
            r"\bmy\s+(?:birthday|birth[\s-]?date|date\s+of\s+birth|bday|b-day|birth\s+year|dob|d\.o\.b)\b",
            r"\b(?:i\s+turned|i'?m\s+turning|i\s+turn)\s+\d{2}\b", rf"\b{_IM}\s+\d{{2}}\s+(?:years\s+old|now)\b",
        ),
        question=(r"\b(?:birthdate|birth\s+date|birthday|date\s+of\s+birth|when\s+was\s+(?:the\s+)?user\s+born|how\s+old|"
                  r"user'?s\s+age)\b",),
        value=_birthdate_value,
        requires_value=True,
    ),
    AttributeSpec(
        "gender", "The user's gender", False,
        statements=(
            r"\bi\s+(?:identify|present)\s+(?:and\s+present\s+)?as\b", r"\bmy\s+pronouns\b", r"\b(?:he/him|she/her|they/them)\b",
            r"\bas\s+a\s+(?:man|woman|guy|girl|male|female|gentleman|lady)\b",
            rf"\b{_IM}\s+(?:a\s+)?(?:man|woman|guy|male|female|girl|lady|gentleman|dude|trans\w*|non-?binary)\b",
        ),
        question=(r"\b(?:gender|pronouns?)\b",),
        value=_gender_value,
        requires_value=True,
    ),
    AttributeSpec(
        "university", "Which university the user attended", False,
        statements=(
            r"\bgraduated\s+(?:from|at)\b",
            r"\b(?:went\s+to|attended|studied\s+at|enrolled\s+at|alum(?:nus|na|ni)?\s+of|degree\s+from|got\s+into|"
            r"(?:bachelor'?s?|master'?s?|ph\.?\s?d|doctorate|mba|b\.?s\.?|m\.?s\.?|b\.?a\.?|m\.?a\.?)\s+(?:\w+\s+){0,4}"
            r"(?:from|at))\b[^.]{0,60}\b(?:university|college|institute|school|academy|polytechnic|(?-i:[A-Z]{2,6}))\b",
            r"\bmy\s+(?:university|college|alma\s+mater|uni)\b",
            r"\b(?:studied|majored|graduated|degree|enrolled)\b[^.]{0,50}\bat\s+(?:the\s+)?(?-i:[A-Z])",
            r"\b(?:student|classes|class|lectures|studying|studies|undergrad|campus|semester|dorm)\s+(?:\w+\s+){0,2}"
            r"(?:at|in)\s+(?:the\s+)?(?-i:[A-Z][\w.'-]*\s+(?:[A-Z][\w.'-]*\s+){0,3}(?:University|College|Institute)|"
            r"(?:University|College|Institute)\s+of\s+[A-Z])",
            r"(?-i:\b(?:University|College|Institute)\b)\s*\((?:mba|b\.?\s?s\.?|b\.?\s?a\.?|m\.?\s?s\.?|m\.?\s?a\.?|"
            r"ph\.?\s?d\.?|bachelor'?s?|master'?s?|bsc|msc)\)",
        ),
        question=(r"\b(?:university|universities|college|attend\w*|alma\s+mater|school)\b",),
        value=_institution_value,
        requires_value=True,
    ),
    AttributeSpec(
        "major", "The user's education major", False,
        statements=(
            r"\b(?:majored|major(?:ing)?)\s+in\b", r"\bmy\s+(?:college\s+|university\s+)?major\b",
            r"\bstudied\s+(?!at\b|abroad\b|for\b|hard\b|up\b|the\b|it\b|with\b)\w+",
            r"\b(?:degree|coursework|diploma)\s+in\b",
            r"\bbackground\s+in\s+(?!my\b|the\b|a\b|an\b|his\b|her\b|our\b|their\b|your\b|mind\b)",
            r"\bmy\s+field\s+of\s+study\b",
            r"\b(?:mba|b\.?s\.?|b\.?a\.?|m\.?s\.?|m\.?a\.?|bsc|msc|ph\.?d\.?|bachelor'?s?|master'?s?|doctorate)\s+"
            r"(?:degree\s+)?in\b",
        ),
        question=(r"\bmajor\b", r"\bfield\s+of\s+study\b", r"\bwhat\s+did\s+(?:the\s+)?user\s+study\b"),
        value=_major_value,
    ),
    AttributeSpec(
        "degree", "The user's highest degree", False,
        statements=(
            r"\b(?:bachelor'?s?|master'?s?|ph\.?\s?d|doctorate|doctoral|associate'?s?\s+degree|mba|m\.?sc|b\.?sc|"
            r"high\s+school\s+diploma)\b",
            r"\bmy\s+(?:highest\s+)?degree\b", r"\b(?:undergrad|graduate)\s+degree\b",
        ),
        question=(r"\bdegree\b", r"\beducational\s+(?:level|attainment)\b"),
        value=_degree_value,
        requires_value=True,
    ),
    AttributeSpec(
        "siblings", "The user's siblings", False,
        statements=(
            r"\bmy\s+(?:\w+\s+){0,2}(?:brothers?|sisters?|siblings?|twin)\b", r"\bonly\s+child\b",
            r"\bno\s+(?:brothers|sisters|siblings)\b",
            r"\b(?:have|had|got)\s+(?:(?:one|two|three|four|five|\d|a|an|any)\s+)?(?:older\s+|younger\s+|little\s+|big\s+)?"
            r"(?:siblings?|brothers?|sisters?)\b",
            r"\b(?:grew\s+up|growing\s+up)\s+with\s+(?:\w+\s+){0,2}(?:siblings?|brothers?|sisters?)\b",
        ),
        question=(r"\b(?:siblings?|brothers?|sisters?)\b",),
        value=_family_value("siblings"),
    ),
    AttributeSpec(
        "mother", "The user's mother", False,
        statements=(r"\bmy\s+(?:\w+\s+)?(?:mom|mother|mum|mama|mommy)\b", r"\b(?:mother|mom|mum|maternal)\b",
                    r"\braised\s+(?:solely|only|entirely)\s+by\s+(?:my\s+)?(?:father|dad)\b"),
        question=(r"\b(?:mother|mom|mum)\b",),
        value=_family_value("mother"),
    ),
    AttributeSpec(
        "father", "The user's father", False,
        statements=(r"\bmy\s+(?:\w+\s+)?(?:dad|father|papa|daddy|pop)\b", r"\b(?:father|dad|paternal)\b",
                    r"\braised\s+(?:solely|only|entirely)\s+by\s+(?:my\s+)?(?:mother|mom|mum)\b"),
        question=(r"\b(?:father|dad)\b",),
        value=_family_value("father"),
    ),
)

# Questions about a family member go to that member, not to the attribute they name ("mother's birth date").
_FAMILY = ("siblings", "mother", "father")
_EXCLUSIVE = (("major", ("university", "degree")), ("degree", ("university",)), ("birthplace", ("residence",)),
              ("birthdate", ("children",)), ("residence", ("birthplace",)))


@dataclass(frozen=True)
class ProfileStatement:
    """One sentence in which the user stated something about an attribute."""

    attribute: str
    text: str
    unit_id: str
    date: Optional[str]
    order: Tuple
    past: bool = False
    values: FrozenSet[str] = frozenset()


@dataclass(frozen=True)
class ResolvedStatement:
    statement: ProfileStatement
    status: str  # current | superseded | established | consistent | conflicting | later


class ProfileLedger:
    """Per-attribute timelines of the user's own statements, resolved by attribute mutability."""

    def __init__(self, attributes: Optional[Sequence[AttributeSpec]] = None) -> None:
        self.specs: Dict[str, AttributeSpec] = {a.name: a for a in (attributes or DEFAULT_ATTRIBUTES)}
        self._statement_re = {a.name: re.compile("|".join(f"(?:{p})" for p in a.statements), re.I)
                              for a in self.specs.values()}
        self._question_re = {a.name: re.compile("|".join(f"(?:{p})" for p in a.question), re.I)
                             for a in self.specs.values()}
        self._statements: Dict[str, List[ProfileStatement]] = {name: [] for name in self.specs}
        self._sorted = True
        self.latest_order: Optional[Tuple] = None
        self.latest_date: Optional[str] = None

    # -- ingest -------------------------------------------------------------------------------------------------
    def add(self, unit_id: str, text: str, *, date: Optional[str] = None, order: Tuple = (),
            speaker: str = "user") -> List[ProfileStatement]:
        """Record the statements in one turn. Only the user's own turns are read."""
        if order and (self.latest_order is None or tuple(order) > self.latest_order):
            self.latest_order, self.latest_date = tuple(order), date
        if str(speaker).lower() != "user":
            return []
        found: List[ProfileStatement] = []
        for si, sentence in enumerate(split_sentences(text)):
            first_person = bool(_FIRST_PERSON.search(sentence))
            for name, spec in self.specs.items():
                if spec.first_person and not first_person:
                    continue
                match = None
                for candidate in self._statement_re[name].finditer(sentence):
                    if name in _FAMILY:
                        # Family statements skip the subject test (they are rarely first-person), except that a kinship
                        # noun owned by someone else is theirs: "Claire lives with me ... her father has weekends".
                        if not candidate.group(0).lower().startswith("my"):
                            before = sentence[:candidate.start()]
                            if _OTHERS_POSSESSIVE.search(before) or (_THE_KIN.search(before) and _OTHER_PERSON.search(sentence)):
                                continue
                        match = candidate
                        break
                    if not _about_someone_else(sentence, candidate.start()):
                        match = candidate
                        break
                if match is None:
                    continue
                values = spec.value(sentence, match.start()) if spec.value else frozenset()
                if spec.requires_value and not values:
                    continue
                near = sentence[max(0, match.start() - 40):match.end() + 40]
                found.append(ProfileStatement(name, sentence, unit_id, date, tuple(order) + (si,),
                                              bool(_PAST.search(near)), values))
        for statement in found:
            self._statements[statement.attribute].append(statement)
        if found:
            self._sorted = False
        return found

    def statements(self, attribute: str) -> List[ProfileStatement]:
        if not self._sorted:
            for timeline in self._statements.values():
                timeline.sort(key=lambda s: s.order)
            self._sorted = True
        return list(self._statements.get(attribute, ()))

    # -- query --------------------------------------------------------------------------------------------------
    def route(self, question: str) -> List[str]:
        """The attributes a question asks about, or none (conditional and preference questions are not routed)."""
        if not question or _CONDITIONAL_QUESTION.search(question):
            return []
        hits = [name for name in self.specs if self._question_re[name].search(question)]
        family = [name for name in hits if name in _FAMILY]
        if family:
            return family
        if "name" in hits and len(hits) > 1:
            hits = [h for h in hits if h != "name"]
        for winner, losers in _EXCLUSIVE:
            if winner in hits:
                hits = [h for h in hits if h not in losers]
        return hits

    def resolve(self, attribute: str) -> List[ResolvedStatement]:
        spec = self.specs[attribute]
        timeline = self.statements(attribute)
        if not timeline:
            return []
        if spec.mutable:
            present = [s for s in timeline if not s.past] or timeline
            last_session = present[-1].order[:1]
            out = []
            for s in timeline:
                current = (not s.past) and s.order[:1] == last_session
                out.append(ResolvedStatement(s, "current" if current else "superseded"))
            return out
        first = timeline[0]
        out = [ResolvedStatement(first, "established")]
        for s in timeline[1:]:
            verdict = _values_conflict(attribute, first.values, s.values, s.text)
            out.append(ResolvedStatement(s, "later" if verdict is None else ("conflicting" if verdict else "consistent")))
        return out

    def render(self, attributes: Iterable[str], *, max_statements: int = 6, max_words: int = 48,
               detail: Optional[str] = None) -> str:
        """The reader-facing ledger for the given attributes; empty when none has a statement."""
        attributes = [a for a in attributes if a in self.specs]
        if len(attributes) > 1 and self.latest_order is not None:
            # Several attributes match ("work status" can mean workload or employment): when some of them were
            # stated in the most recent session, those are what the question is about.
            recent = [a for a in attributes
                      if any(s.order[:1] == self.latest_order[:1] for s in self.statements(a))]
            if recent:
                attributes = recent
        blocks: List[str] = []
        for attribute in attributes:
            spec = self.specs[attribute]
            resolved = self.resolve(attribute)
            if detail and not spec.mutable:
                resolved = _focus(resolved, detail)
            if not resolved:
                continue
            shown = _select(resolved, spec.mutable, max_statements)
            if spec.mutable:
                head = f"{spec.label} (changes over time; the latest statement is the current one):"
            else:
                head = (f"{spec.label} (does not change over time; the user's first statement is the established "
                        f"record, and a later claim that differs does not replace it):")
            lines = [head]
            if detail == "employment" and attribute == "career":
                status = _employment_line(resolved)
                if status:
                    lines.append(status)
            for r in shown:
                lines.append(f"- {r.statement.date or 'undated'}, {self._label(r)}: \"{_clip(r.statement.text, max_words)}\"")
            blocks.append("\n".join(lines))
        if not blocks:
            return ""
        return ("Profile ledger (Telarch): the user's own statements about the asked attribute, oldest first.\n"
                + "\n".join(blocks))

    def _label(self, r: ResolvedStatement) -> str:
        recent = (self.latest_order is not None and r.statement.order[:1] == self.latest_order[:1])
        if r.status == "current":
            return "CURRENT, stated in the most recent session" if recent else "CURRENT (latest statement)"
        if r.status == "superseded":
            return "earlier, superseded"
        if r.status == "established":
            return "ESTABLISHED RECORD (first statement)"
        if r.status == "consistent":
            return "later, matches the record"
        if r.status == "conflicting":
            return "LATER CONFLICTING CLAIM (contradicts the established record; the record stands)"
        return "later claim (where it differs, the established record stands)"


_JOB_TITLE = (r"(?:intern|manager|director|engineer|analyst|consultant|designer|developer|specialist|coordinator|"
              r"assistant|associate|officer|lead|senior|junior|executive|teacher|nurse|accountant|lawyer|clerk|"
              r"technician|administrator|supervisor|representative)")
# The working arrangement a career statement describes. Only explicit wording counts: someone "looking for a new
# job", "job hunting" or doing some "freelance work" may still be employed, so none of these sets the status.
_EMPLOYMENT_KINDS: Tuple[Tuple[str, "re.Pattern[str]"], ...] = (
    ("self-employed or own business", re.compile(
        r"\b(?:self[- ]employed|entrepreneur\w*|(?:co-?)?found(?:ed|ing)\s+(?:a|my|our|the)\b|own\s+boss|"
        r"(?:" + _IM + r"|" + _IVE + r"\s+been)\s+" + _ADV + r"(?:an?\s+)?freelanc\w*|i\s+" + _ADV + r"freelance|"
        r"(?:went|gone|going|go)\s+freelance|"
        r"my\s+own\s+(?:\w+\s+)?(?:business|company|firm|startup|shop|practice|studio|agency|consultancy)|"
        r"(?:started|starting|launched|launching|opened|opening|running|run|runs)\s+(?:a|my|our)\s+(?:own\s+)?(?:\w+\s+)?"
        r"(?:business|company|startup|firm|shop|practice|studio|agency|consultancy))\b", re.I)),
    ("unemployed", re.compile(
        r"\b(?:unemployed|laid\s+off|(?:was|been|got|getting)\s+let\s+go|(?:got|was|been|getting)\s+fired|"
        r"lost\s+(?:my|the)\s+(?:\w+\s+){0,2}(?:job|position)|between\s+jobs|out\s+of\s+(?:a\s+)?(?:work|job)|"
        r"(?:not|no\s+longer)\s+(?:currently\s+)?employed|quit\s+my\s+job|resigned\s+from)\b", re.I)),
    ("retired", re.compile(
        r"\b(?:i\s+(?:just\s+|recently\s+|finally\s+|officially\s+)?retired|(?:i'?m|i\u2019m|i\s+am)\s+(?:now\s+|"
        r"officially\s+|happily\s+)?retired|since\s+i\s+retired)\b", re.I)),
    ("employed", re.compile(
        r"\b(?:work(?:s|ed|ing)?\s+(?:as|at|for)|employed\s+(?:as|at|by|full[- ]time|part[- ]time)|"
        + _IM + r"\s+" + _ADV + r"employed|"
        r"(?:started|starting|landed|accepted|took|got|began|joined|joining|start)\s+(?:a\s+|an\s+|my\s+|the\s+)?"
        r"(?:new\s+)?(?:\w+\s+){0,2}(?:job|role|position|internship|post)|(?:joined|joining|started\s+at|now\s+at)\s+"
        r"(?-i:[A-Z])|promot(?:ed|ion)|(?:new|current)\s+(?:job|role|position|employer|title)|"
        r"my\s+(?:job|role|position|title|employer)\s+(?:is|as|at|now)|as\s+(?:an?|the)\s+(?:\w+\s+){0,4}" + _JOB_TITLE +
        r"|" + _IM + r"\s+" + _ADV + r"(?:an?|the)\s+(?:\w+\s+){0,4}" + _JOB_TITLE + r")\b", re.I)),
)


def _employment_kind(text: str) -> Optional[str]:
    """The working arrangement a statement describes; when it names several, the one mentioned last ("I got laid
    off but landed a new job" is employed, "I quit my job to start my own company" is self-employed)."""
    best: Optional[Tuple[int, str]] = None
    for kind, pattern in _EMPLOYMENT_KINDS:
        for m in pattern.finditer(text):
            if best is None or m.start() > best[0]:
                best = (m.start(), kind)
    return best[1] if best else None


def _employment_line(resolved: List[ResolvedStatement]) -> Optional[str]:
    """For a question about employment status: the status the career statements imply before and after the latest
    change (the most recent session that implies one, against the last statement before it).

    A reader shown a timeline of job titles tends to read a new title as a change of employment status; this line
    separates the two. It compares the latest change only: an earlier spell out of work does not make a later move
    between two jobs a change of status."""
    marked = []
    for r in resolved:
        kind = _employment_kind(r.statement.text)
        if kind is not None:
            marked.append((r.statement, kind))
    if not marked:
        return None
    head = "Employment status implied by these statements (employed, unemployed, self-employed or own business, retired):"
    note = ("A new job, title or employer while working does not change the employment status; only a move between "
            "these states does.")
    last_session = marked[-1][0].order[:1]
    now_statement, now_kind = marked[-1]
    before = [m for m in marked if m[0].order[:1] != last_session]
    if not before:
        return f"{head} {now_kind} ({now_statement.date or 'undated'}); no earlier statement implies one. {note}"
    prev_statement, prev_kind = before[-1]
    verdict = "unchanged" if prev_kind == now_kind else "changed"
    return (f"{head} before the latest change, {prev_kind} ({prev_statement.date or 'undated'}); after it, {now_kind} "
            f"({now_statement.date or 'undated'}); {verdict}. {note}")


def _focus(resolved: List[ResolvedStatement], detail: str) -> List[ResolvedStatement]:
    """For a question about one detail of a family member (a date, a name), keep statements that carry it."""
    want = {"date": lambda s: bool(_years(s.text)), "name": lambda s: bool(_proper_nouns(s.text)),
            "kind": lambda s: bool(_SIBLING_KIND.search(s.text)), "count": lambda s: bool(_SIBLING_COUNT.search(s.text))
            or bool(_HAS_NONE["siblings"].search(s.text))}.get(detail)
    if want is None:
        return resolved
    kept = [r.statement for r in resolved if want(r.statement)]
    if not kept:
        return resolved
    first = kept[0]
    out = [ResolvedStatement(first, "established")]
    for s in kept[1:]:
        verdict = _values_conflict(first.attribute, first.values, s.values, s.text)
        out.append(ResolvedStatement(s, "later" if verdict is None else ("conflicting" if verdict else "consistent")))
    return out


def question_detail(question: str) -> Optional[str]:
    """Which detail a question asks for, if any: the user's employment status ("employment"), or a date, name, kind
    or count of a family member."""
    q = question.lower()
    if re.search(r"\b(?:employment\s+(?:status|situation)|(?:un|self[- ])?employed)\b", q):
        return "employment"
    if re.search(r"\b(?:birth\s*date|birthday|born|date\s+of\s+birth|how\s+old|age)\b", q):
        return "date"
    if re.search(r"\bname\b", q):
        return "name"
    if re.search(r"\b(?:type|relation|relationship|brother\s+or\s+sister|gender)\b", q):
        return "kind"
    if re.search(r"\b(?:how\s+many|number\s+of|count)\b", q):
        return "count"
    return None


_QUESTION_LIKE = re.compile(
    r"\?\s*$|^\W*(?:how|what|why|when|where|which|who|whom|should|could|can|would|do|does|did|is|are|will|any)\b", re.I)
_ANNOUNCE = re.compile(
    r"\b(?:update|now|currently|recently|just|started|starting|moved|officially|switched|became|got|left|joined|"
    r"quit|lost|landed|took|retired|i'?m\s+(?:a|an|the|in|at|on|with|still|no)|i\s+(?:work|live|am)|"
    r"i\u2019m\s+(?:a|an|the|in|at|on|with|still|no))\b", re.I)


def _earlier_rank(r: ResolvedStatement) -> Tuple[bool, bool, Tuple]:
    """Earlier statements that state the user's situation outrank the user's own questions."""
    text = r.statement.text
    return (not _QUESTION_LIKE.search(text), bool(_ANNOUNCE.search(text)), r.statement.order)


def _select(resolved: List[ResolvedStatement], mutable: bool, limit: int) -> List[ResolvedStatement]:
    if len(resolved) <= limit:
        return resolved
    if mutable:
        # Keep the latest current statements and the earlier statements that best describe the previous state, so a
        # "how did it change" question sees both ends: at least two earlier slots when there are earlier statements.
        current = [r for r in resolved if r.status == "current"]
        earlier = [r for r in resolved if r.status != "current"]
        room = min(len(earlier), max(limit - len(current), min(2, limit // 3)))
        current = sorted(current, key=_earlier_rank, reverse=True)[:limit - room] if limit > room else []
        kept = sorted(earlier, key=_earlier_rank, reverse=True)[:room]
        return sorted(kept + current, key=lambda r: r.statement.order)
    established = resolved[0]
    rest = resolved[1:]
    conflicting = [r for r in rest if r.status == "conflicting"]
    others = [r for r in rest if r.status != "conflicting"]
    room = limit - 1
    keep = conflicting[-room:] if len(conflicting) >= room else conflicting + others[:room - len(conflicting)]
    return [established] + sorted(keep, key=lambda r: r.statement.order)


def _clip(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.replace('"', "'")
    return (" ".join(words[:max_words]) + " ...").replace('"', "'")


__all__ = [
    "AttributeSpec",
    "DEFAULT_ATTRIBUTES",
    "ProfileLedger",
    "ProfileStatement",
    "ResolvedStatement",
    "question_detail",
    "split_sentences",
]
