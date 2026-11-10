"""severity: low
finding: BehavioralGovernor.evaluate matches a revoked history item only by exact lowercase "revoked", so a history directive marked "Revoked" is approved while the same marking on an invariant is honoured

evaluate (src/harpe/governor.py:80-176) reads ``epistemic_state`` twice. The invariant rule lower-cases it
(governor.py:139, ``"revoked" in str(...).lower()``), so "REVOKED" or "Revoked" counts. The history rule compares it with
``hist_state == "revoked"`` exactly (governor.py:147-148), so a candidate-history directive whose state is "Revoked" (a store that
capitalises its states) raises no contradiction and the action comes back APPROVED with "All operational
invariants satisfied." A revoked directive replayed through history is supposed to trigger review (the
module's "screens out contradictory or revoked historical directives", governor.py:6-7). Low: the module is
not exported from top-level harpe and is documented as heuristic screening only (governor.py:9-11).
"""
from __future__ import annotations

from harpe import ContextItem
from harpe.governor import BehavioralGovernor, GovernorVerdict


def _history(state: str) -> ContextItem:
    return ContextItem(id="old-host", content="Deploy to the old staging host 10.0.0.5.",
                       metadata={"epistemic_state": state})


def test_control_lowercase_revoked_history_needs_review():
    decision = BehavioralGovernor().evaluate("deploy build 42 to staging", [], candidate_history=[_history("revoked")])
    assert decision.verdict == GovernorVerdict.NEEDS_REVIEW


def test_control_capitalised_revoked_invariant_is_honoured():
    inv = ContextItem(id="inv", content="Old endpoint", dimension="invariant",
                      metadata={"epistemic_state": "Revoked", "key": "10.0.0.5"})
    decision = BehavioralGovernor().evaluate("curl http://10.0.0.5/api", [inv])
    assert decision.verdict == GovernorVerdict.SUPPRESSED


def test_control_active_history_is_approved():
    decision = BehavioralGovernor().evaluate("deploy build 42 to staging", [], candidate_history=[_history("active")])
    assert decision.verdict == GovernorVerdict.APPROVED


def test_a_capitalised_revoked_history_item_is_not_approved():
    decision = BehavioralGovernor().evaluate("deploy build 42 to staging", [], candidate_history=[_history("Revoked")])
    assert decision.verdict == GovernorVerdict.NEEDS_REVIEW, decision.verdict
