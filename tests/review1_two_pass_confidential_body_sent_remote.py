"""severity: high
finding: TwoPassBuild swaps a confidential body (or places a confidential description) into a context whose Pass 1 was built for the remote destination, which withholds every confidential item

ContextBudgetEngine.synthesize defaults to destination="remote" and "withholds every item marked confidential,
pinned or not, before any budgeting, so it cannot take a budget slot or appear in the output"
(src/harpe/engine.py:476, 489-491, 521-523). TwoPassBuild (src/harpe/two_pass.py:103-160) continues that build:
it checks that each caller-supplied body or description has the row's id and is not a tier-0 item
(two_pass.py:140-151), but never looks at ``sensitivity``. ``request`` (two_pass.py:175-191) then swaps a body
marked confidential into the context and ``items()`` (two_pass.py:225-234) hands it out; in ``descriptions=``
mode a confidential description is in the context from the start. The build's receipt still lists the Pass-1
withheld ids as dropped, so the ledger records that confidential items were withheld while one was sent.

The tests accept any refusal: an exception when the build is made or the row is requested, or ``request``
returning False, as long as no confidential item ends up in ``items()``.
"""
from __future__ import annotations

from harpe import ContextBudgetEngine, ContextItem, TokenBudget
from harpe.two_pass import TwoPassBuild


def _budget() -> TokenBudget:
    return TokenBudget(total_budget=1000, reserved_completion=0, reserved_tools=0, safety_margin=0)


def _pass1(rows):
    policy = ContextItem(id="policy", content="Answer briefly.", pinned=True, token_count=10)
    withheld = ContextItem(id="vault", content="vault entry", token_count=10, sensitivity="confidential")
    report = ContextBudgetEngine().synthesize([policy, *rows, withheld], _budget(), destination="remote")
    assert [it.id for it in report.withheld_items] == ["vault"]
    return report


def _no_confidential(build) -> None:
    sent = build.items()
    assert [it.id for it in sent if it.sensitivity == "confidential"] == [], "a confidential item is in the context"
    assert all("hunter2" not in it.content for it in sent)


def test_control_a_normal_body_expands_and_pass1_withholds_confidential_items():
    report = _pass1([ContextItem(id="a", content="a: incident notes (title)", token_count=10)])
    body = ContextItem(id="a", content="a: full incident notes", token_count=50)
    build = TwoPassBuild(report, bodies={"a": body}, budget_tokens=1000)
    assert build.request("a") is True
    assert [(it.id, it.content) for it in build.items()] == [("policy", "Answer briefly."), ("a", body.content)]
    assert "vault" in build.dropped_ids


def test_a_confidential_body_is_not_swapped_into_a_remote_context():
    report = _pass1([ContextItem(id="a", content="a: incident notes (title)", token_count=10)])
    body = ContextItem(id="a", content="a: full incident notes, root password hunter2", token_count=50,
                       sensitivity="confidential")
    try:
        build = TwoPassBuild(report, bodies={"a": body}, budget_tokens=1000)
    except (ValueError, TypeError):
        return  # refusing the confidential body up front is correct
    try:
        build.request("a")
    except Exception:
        pass  # refusing the request by raising is correct too
    _no_confidential(build)


def test_a_confidential_description_is_not_placed_in_a_remote_context():
    report = _pass1([ContextItem(id="a", content="a: full incident notes", token_count=50)])
    desc = ContextItem(id="a", content="a: notes mention hunter2", token_count=10, sensitivity="confidential")
    try:
        build = TwoPassBuild(report, descriptions={"a": desc}, budget_tokens=1000)
    except (ValueError, TypeError):
        return
    _no_confidential(build)
