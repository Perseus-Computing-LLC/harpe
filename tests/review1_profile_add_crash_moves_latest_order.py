"""severity: low
finding: ProfileLedger.add() moves latest_order/latest_date (and the arrival counter) before it parses the turn, so a turn whose ingest crashes still becomes "the most recent session" though none of its statements were recorded

src/harpe/profile.py:670-675 updates ``_arrivals``, ``latest_order`` and ``latest_date`` first. Only after that does
add() split and match the turn (:679-705) and append what it found (:706-709). The statement part is all-or-nothing,
because statements are gathered in ``found`` and appended at the end. The "latest" bookkeeping is not. If add()
dies anywhere in the parse (a kill, a raising custom ``AttributeSpec.value``, a bytes ``text``), the ledger already
names that turn's session as the most recent one, although nothing from it is in any timeline. ``render`` and
``_label`` (:759-765, :793-796) read ``latest_order``. The earlier Denver statement loses its "CURRENT, stated in the
most recent session" label and is shown as merely "CURRENT (latest statement)". ``latest_date`` reports a date the
ledger holds no statement for.

The crash is simulated by making ``split_sentences`` raise, which is the first step after the bookkeeping.
"""
from __future__ import annotations

import pytest

import harpe.profile as profile_mod
from harpe.profile import ProfileLedger


def _crash(*_a, **_k):
    raise RuntimeError("simulated kill during ingest")


def test_control_a_recorded_turn_is_the_most_recent_session():
    ledger = ProfileLedger()
    found = ledger.add("u1", "I live in Denver now.", date="2026-01-01", order=(1, 0))
    assert [s.attribute for s in found] == ["residence"]
    assert ledger.latest_order == (1, 0) and ledger.latest_date == "2026-01-01"
    assert "CURRENT, stated in the most recent session" in ledger.render(["residence"])


def test_a_crashed_add_leaves_the_latest_session_where_it_was(monkeypatch):
    ledger = ProfileLedger()
    ledger.add("u1", "I live in Denver now.", date="2026-01-01", order=(1, 0))
    before = (ledger.latest_order, ledger.latest_date, ledger.render(["residence"]))

    monkeypatch.setattr(profile_mod, "split_sentences", _crash)
    with pytest.raises(RuntimeError):
        ledger.add("u5", "I moved to Seattle.", date="2026-05-01", order=(5, 0))
    monkeypatch.undo()

    assert ledger.statements("residence")[-1].unit_id == "u1", "precondition: nothing from u5 was recorded"
    after = (ledger.latest_order, ledger.latest_date, ledger.render(["residence"]))
    assert after == before, (
        "a turn whose ingest failed became the ledger's most recent session: "
        f"latest_order={ledger.latest_order}, latest_date={ledger.latest_date}"
    )
