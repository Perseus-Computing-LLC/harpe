"""severity: medium
finding: ProfileLedger.add with the default order=() orders a timeline by sentence position inside each turn, so a later claim displaces the established record and an older residence is rendered CURRENT

ProfileLedger.add (src/harpe/profile.py:659-698) takes ``order: Tuple = ()`` and stores each statement's order as
``tuple(order) + (si,)`` where ``si`` is the sentence's index inside its own turn (profile.py:692).
``statements`` sorts each timeline by that key (profile.py:700-705), and ``resolve`` treats ``order[:1]`` as the
session (profile.py:723-741). With the default ``order=()`` the key is just ``(si,)``: arrival order is lost and
the "session" is the sentence index. A claim made later in the conversation, in a turn's first sentence, sorts
before a claim made earlier in a turn's second sentence. CHANGELOG 0.1.0-alpha.8 (CHANGELOG.md:179-183) promises that for
attributes that do not change "the first statement is the established record and a later claim with a different value is
marked as conflicting", and for attributes that change "statements from the latest session are current"; render() says "oldest
first" and "the latest statement is the current one". With the default order, the later MIT claim becomes the
established university record and the earlier Long Beach statement is marked conflicting; the earlier Denver
residence is rendered "CURRENT (latest statement)" and the later move to Seattle is superseded.
"""
from __future__ import annotations

from harpe import ProfileLedger

UNIVERSITY = [
    ("t0", "Hello again. I graduated from California State University, Long Beach, so I know the area."),
    ("t1", "Since I studied at MIT, I like to problem-solve on the spot."),
]
RESIDENCE = [
    ("t0", "Quick update. I live in Denver with my two cats."),
    ("t1", "Big news: I just moved to Seattle for the new job."),
]


def _ledger(turns, orders=None) -> ProfileLedger:
    ledger = ProfileLedger()
    for i, (unit_id, text) in enumerate(turns):
        if orders is None:
            ledger.add(unit_id, text)
        else:
            ledger.add(unit_id, text, order=orders[i])
    return ledger


def _status(ledger: ProfileLedger, attribute: str):
    return {r.statement.unit_id: r.status for r in ledger.resolve(attribute)}


def _timeline(ledger: ProfileLedger, attribute: str):
    return [r.statement.unit_id for r in ledger.resolve(attribute)]


def test_control_explicit_order_keeps_the_first_statement_as_the_record():
    uni = _ledger(UNIVERSITY, orders=[(0, 0), (3, 1)])
    assert _timeline(uni, "university") == ["t0", "t1"]
    assert _status(uni, "university") == {"t0": "established", "t1": "conflicting"}
    res = _ledger(RESIDENCE, orders=[(0, 0), (1, 1)])
    assert _status(res, "residence") == {"t0": "superseded", "t1": "current"}


def test_control_default_order_with_one_sentence_turns_keeps_arrival_order():
    uni = _ledger([("t0", UNIVERSITY[0][1].split(". ", 1)[1]), UNIVERSITY[1]])
    assert _timeline(uni, "university") == ["t0", "t1"]
    assert _status(uni, "university")["t0"] == "established"


def test_default_order_the_first_statement_stays_the_established_record():
    uni = _ledger(UNIVERSITY)
    assert _timeline(uni, "university") == ["t0", "t1"], "the timeline is not in arrival order"
    status = _status(uni, "university")
    assert status["t0"] == "established", f"a later claim displaced the established record: {status}"
    assert status["t1"] != "established"


def test_default_order_the_latest_residence_statement_is_current():
    res = _ledger(RESIDENCE)
    assert _timeline(res, "residence") == ["t0", "t1"], "the timeline is not in arrival order"
    assert _status(res, "residence")["t1"] == "current", "the user's latest move is marked superseded"
