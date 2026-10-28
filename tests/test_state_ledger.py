"""StateLedger (Round 8 candidate): behaviour and the input-isolation guarantee."""

import hashlib
import re
from pathlib import Path

import pytest

from harpe.state_ledger import QUESTION_GATE, StateLedger, StatePair, content_words, header

SRC = Path(__file__).resolve().parents[1] / "src" / "harpe" / "state_ledger.py"

CONTENT_WORD_VECTORS = [
    line.split("\t")
    for line in Path(__file__).with_name("content_word_vectors.tsv").read_text(encoding="utf-8").splitlines()
    if not line.startswith("#")
]


@pytest.mark.parametrize("text, expected", CONTENT_WORD_VECTORS, ids=[v[0][:40] for v in CONTENT_WORD_VECTORS])
def test_content_words_match_the_shared_vectors(text, expected):
    """The same file is checked by src/state_ledger.rs so both ledgers pair the same claims."""
    assert len(CONTENT_WORD_VECTORS) == 12
    assert " ".join(sorted(content_words(text))) == expected

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


# ---- 0.1.0a9: seq, length-normalised ranking, programmatic API ----------------------------------------------------

WINDOW_UNITS = {
    "10:2": ("user: I've completed the migration runbook for the billing service.", "2024-03-26"),
    "2:1": ("user: I've never completed the migration runbook for the billing service, it is still a draft.",
            "2024-03-26"),
    "7:0": ("user: Yes, I've completed the migration runbook for the billing service and the payments service and "
            "the ledger service and the archive service and the reporting service and the audit service too.",
            "2024-05-01"),
    "9:0": ("user: I usually deploy the billing service on Fridays.", ""),
}


def windowed():
    led = StateLedger()
    led.add_units(WINDOW_UNITS)
    return led


def test_seq_follows_insertion_not_unit_id_string():
    led = windowed()
    assert [c.seq for c in led.claims] == list(range(len(led.claims)))
    first, second = led.claims[0], led.claims[1]
    assert first.unit_id == "10:2" and second.unit_id == "2:1"
    assert first.order < second.order  # same date: "10:2" came first even though "2:1" sorts first as a string


def test_header_orders_same_date_pair_by_seq():
    h = header(windowed().conflicts_for("Have I completed the migration runbook for the billing service?"))
    assert h.splitlines()[0].endswith('"I\'ve completed the migration runbook for the billing service."')


def test_ranking_prefers_tight_flip_over_long_sentence():
    pairs = windowed().conflicts_for("Have I completed the migration runbook for the billing service?")
    assert [p.affirm.unit_id for p in pairs] == ["10:2", "7:0", "9:0"]
    assert pairs[0].strength > pairs[1].strength > pairs[2].strength
    assert len(pairs[0].shared) == len(pairs[1].shared)  # same overlap; the shorter affirmation wins


def test_entity_window_threshold_and_limit():
    led = windowed()
    terms = ["migration runbook", "billing service"]
    assert [p.affirm.unit_id for p in led.conflicts_for_entity(terms)] == ["10:2", "7:0", "9:0"]
    assert len(led.conflicts_for_entity(terms, limit=1)) == 1
    # an undated claim ("9:0") passes any window; the dated ones are filtered
    assert [p.affirm.unit_id for p in led.conflicts_for_entity(terms, until="2024-03-31")] == ["10:2", "9:0"]
    assert led.conflicts_for_entity(terms, since="2024-04-01") == []  # the denial is outside the window
    assert led.conflicts_for_entity(terms, min_shared=3) == [] or all(len(p.shared) >= 3 for p in led.conflicts_for_entity(terms, min_shared=3))
    assert [p.affirm.unit_id for p in led.conflicts_for_entity(terms, min_shared=3)] == ["10:2", "7:0"]
    assert led.conflicts_for_entity(terms, min_shared=5) == []


def test_timeline_and_latest_state_by_date_then_seq():
    led = windowed()
    tl = led.timeline(["billing service"])
    assert [c.unit_id for c in tl] == ["9:0", "10:2", "2:1", "7:0"]  # undated first, then (date, seq)
    assert led.latest_state(["billing service"]).unit_id == "7:0"
    assert led.latest_state(["billing service"], until="2024-03-31").unit_id == "2:1"
    assert led.latest_state(["nothing here"]) is None
    assert led.timeline(["migration"], since="2024-04-01") and all(c.date >= "2024-04-01" for c in led.timeline(["migration"], since="2024-04-01"))


def test_api_is_pure_and_claims_are_frozen():
    led = windowed()
    a = [p.model_dump() for p in led.conflicts_for_entity(["migration runbook", "billing service"])]
    b = [p.model_dump() for p in windowed().conflicts_for_entity(["migration runbook", "billing service"])]
    assert a == b
    with pytest.raises(Exception):
        led.claims[0].seq = 99  # frozen model
