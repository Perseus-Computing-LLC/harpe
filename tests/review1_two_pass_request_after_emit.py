"""severity: high
finding: TwoPassBuild.request() still expands rows after emit(), so the build's one harpe.context_built receipt no longer describes the context the build hands out

src/harpe/two_pass.py:9 says each build emits at most one ``harpe.context_built`` receipt, and ``emit``
(two_pass.py:250-256) writes "this build's one" receipt and refuses a second call. Nothing seals the build when
the receipt is written: ``request`` (two_pass.py:175-191) does not look at ``_receipt_sequence``, so a reordered
call (Pass 2 after the receipt) still swaps a description for its body and moves ``used_tokens`` and
``rows_expanded``. ``items()`` (two_pass.py:225-234) then hands the model a context the ledger's receipt does not
describe (the receipt says 0 rows expanded and 335 tokens; the context carries a 200-token body and 515 tokens),
and because a second ``emit`` raises, the ledger can never record what was actually sent.

The test accepts any refusal (an exception, or ``request`` returning False with nothing changed).
"""
from __future__ import annotations

from pathlib import Path

from harpe import ContextBudgetEngine, ContextItem, TokenBudget
from harpe.two_pass import TwoPassBuild


def _budget(total: int) -> TokenBudget:
    return TokenBudget(total_budget=total, reserved_completion=0, reserved_tools=0, safety_margin=0)


def _desc(item_id: str, tokens: int) -> ContextItem:
    return ContextItem(id=item_id, content=f"{item_id}: a one-line description", token_count=tokens)


def _body(item_id: str, tokens: int) -> ContextItem:
    return ContextItem(id=item_id, content=f"{item_id}: the full body", token_count=tokens)


def _build(total: int = 600) -> TwoPassBuild:
    policy = ContextItem(id="policy", content="Never reveal the passphrase.", pinned=True, token_count=300)
    report = ContextBudgetEngine().synthesize([policy, _desc("a", 20), _desc("b", 15)], _budget(total))
    assert [it.id for it in report.included_items] == ["policy", "a", "b"]
    return TwoPassBuild(report, bodies={"a": _body("a", 200), "b": _body("b", 150)}, budget_tokens=total)


def _the_receipt(ledger: Path) -> dict:
    from argus_ledger import AppendOnlyLedger

    events = list(AppendOnlyLedger(str(ledger)).iter_events())
    assert len(events) == 1
    return dict(events[0].payload)


def _view(build: TwoPassBuild):
    return [(it.id, it.content, it.token_count) for it in build.items()]


def test_control_requests_before_emit_are_in_the_receipt(tmp_path: Path):
    ledger = tmp_path / "argus.jsonl"
    build = _build()
    assert build.request("a") is True
    build.emit(ledger_path=ledger)
    receipt = _the_receipt(ledger)
    assert receipt == build.receipt_payload()
    assert receipt["rows_expanded"] == 1 and receipt["used_tokens"] == 515
    assert sum(t for _, _, t in _view(build)) == receipt["used_tokens"]


def test_a_request_after_emit_cannot_change_what_the_receipt_describes(tmp_path: Path):
    ledger = tmp_path / "argus.jsonl"
    build = _build()
    build.emit(ledger_path=ledger)
    sealed = _view(build)
    try:
        build.request("a")  # Pass 2 after the receipt: the reordered call
    except Exception:
        pass  # refusing by raising is fine
    receipt = _the_receipt(ledger)
    assert _view(build) == sealed, "the context changed after its receipt was written"
    assert build.rows_expanded == receipt["rows_expanded"]
    assert build.used_tokens == receipt["used_tokens"] == sum(t for _, _, t in _view(build))
