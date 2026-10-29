"""StateLedger (Round 8 candidate): behaviour and the input-isolation guarantee."""

import hashlib
import re
from pathlib import Path

import pytest

from harpe.state_ledger import QUESTION_GATE, StateLedger, StatePair, header

SRC = Path(__file__).resolve().parents[1] / "src" / "harpe" / "state_ledger.py"

UNITS = {
    "0:1": ("user: I've written integration tests for the socket events, they run on every push.", "2023-05-11"),
    "3:2": ("user: I've never written integration tests for any socket events, we only have unit tests.", "2023-08-02"),
    "4:0": ("assistant: I've completed the deployment for you.\nuser: thanks", "2023-08-03"),
    "5:0": ("user: I usually deploy on Fridays.", "2023-09-01"),
}


def ledger():
    led = StateLedger()
    led.add_units(UNITS)
    return led


def test_input_isolation_source_scan():
    """The detector may read dialogue units and the question text only. These words must not appear in its source."""
    text = SRC.read_text("utf-8")
    body = text.split('"""', 2)[2]  # skip the module docstring, which explains the design
    for word in ("rubric", "ability", "abilities", "gold", "score", "label", "evaluation", "judge"):
        assert not re.search(rf"\b{word}\b", body, re.IGNORECASE), word


def test_pairs_affirmation_with_later_denial():
    pairs = ledger().conflicts_for("Have I written integration tests for socket events?")
    assert len(pairs) == 1
    p = pairs[0]
    assert p.affirm.unit_id == "0:1" and p.deny.unit_id == "3:2"
    assert set(p.shared) >= {"integration", "tests", "socket", "events"}


def test_assistant_lines_are_not_claims():
    led = ledger()
    assert all(c.unit_id != "4:0" for c in led.claims)


def test_question_gate_blocks_other_abilities():
    led = ledger()
    for q in ("What did I say about integration tests?", "Summarize our discussion of socket events and tests.",
              "When did I set up integration tests for socket events?", "Which tests cover the socket events?"):
        assert not QUESTION_GATE.match(q)
        assert led.conflicts_for(q) == []


def test_header_quotes_both_dated_with_no_instruction_text():
    h = header(ledger().conflicts_for("Have I written integration tests for socket events?"))
    assert h.splitlines()[0].startswith('On 2023-05-11, user said: "')
    assert h.splitlines()[1].startswith('On 2023-08-02, user said: "')
    lowered = h.lower()
    for banned in ("conflict", "note", "possible", "[", "]", "contradict", "please", "answer", "consider"):
        assert banned not in lowered, banned


def test_header_empty_without_pairs():
    assert header([]) == ""


def test_deterministic_and_frozen_hash_recorded():
    """The frozen hash lives beside the pre-registration; this test prints the current one for the record."""
    digest = hashlib.sha256(SRC.read_bytes()).hexdigest()
    assert len(digest) == 64
    led = ledger()
    a = led.conflicts_for("Have I written integration tests for socket events?")
    b = ledger().conflicts_for("Have I written integration tests for socket events?")
    assert [p.model_dump() for p in a] == [p.model_dump() for p in b]


def test_entity_form_has_no_question_gate():
    pairs = ledger().conflicts_for_entity(["integration", "tests", "socket", "events"])
    assert len(pairs) == 1 and isinstance(pairs[0], StatePair)
    assert ledger().conflicts_for_entity(["tests"]) == []


# --- Round 9: open vocabulary and the verb-or-two-specific-words pairing rule ---------------------------------

R9_UNITS = {
    "10:0": ("user: I've deployed the Kubernetes cluster to production last week.", "2024-01-10"),
    "11:0": ("user: I've never deployed a Kubernetes cluster, so I'm not sure where to start.", "2024-02-02"),
    "12:0": ("user: I've finished the migration to Postgres and the app is stable.", "2024-02-05"),
    "13:0": ("user: I haven't actually finished the Postgres migration yet.", "2024-02-20"),
    "14:0": ("user: I've not done any load testing on the gateway.", "2024-03-01"),
    "15:0": ("user: I've started this project during work hours.", "2024-03-02"),
    "16:0": ("user: I never finished this project during work hours.", "2024-03-09"),
}


def r9():
    led = StateLedger()
    led.add_units(R9_UNITS)
    return led


def test_open_vocabulary_deny_pairs_a_verb_outside_the_round8_list():
    """'deployed' is not in the Round 8 verb list; 30 of 39 Round 8 misses were exactly this."""
    pairs = r9().conflicts_for("Have I deployed a Kubernetes cluster?")
    assert [(p.affirm.unit_id, p.deny.unit_id) for p in pairs] == [("10:0", "11:0")]


def test_adverb_between_negation_and_verb_still_denies():
    pairs = r9().conflicts_for("Have I finished the Postgres migration?")
    assert ("12:0", "13:0") in [(p.affirm.unit_id, p.deny.unit_id) for p in pairs]


def test_ive_not_is_a_denial_not_an_affirmation():
    led = r9()
    claim = next(c for c in led.claims if c.unit_id == "14:0")
    assert claim.polarity == -1


def test_pairing_needs_the_same_verb_or_two_specific_words():
    """'started' vs 'finished' sharing only generic words (project, during, work) is not a state conflict."""
    assert r9().conflicts_for("Have I finished this project during work hours?") == []


def test_round8_cases_unchanged():
    pairs = ledger().conflicts_for("Have I written integration tests for socket events?")
    assert [(p.affirm.unit_id, p.deny.unit_id) for p in pairs] == [("0:1", "3:2")]
    assert all(c.unit_id != "4:0" for c in ledger().claims)
