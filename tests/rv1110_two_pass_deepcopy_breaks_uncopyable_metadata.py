"""severity: medium
finding: regression from the round-1 fix: TwoPassBuild raises TypeError when any report item or row carries metadata deepcopy cannot copy (a lock, a handle, a client); the same build worked at 723ff5b and 2a43c31, and no hook is involved

The receipt fix made the build keep "its own copies of the items it is given" (CHANGELOG, Unreleased) with
`model_copy(deep=True)` in `_counted` (src/harpe/two_pass.py:205-214) and again for every item on every `items()`
call (:281). `ContextItem.metadata` is `Dict[str, Any]`, "Arbitrary item metadata" (models.py:99); a report whose
items carry a live object there used to build (the build held the caller's objects) and now raises
`TypeError: cannot pickle '_thread.lock' object` from the constructor. Sibling of
rv1110_hook_deepcopy_breaks_uncopyable_metadata.py (same cause, other entry point, other commit); here every
caller is affected, with or without hooks.

What the receipt needs is that the fields it accounts for (id, content, token_count, pinned/priority/dimension,
sensitivity) cannot change under it; that does not require cloning opaque metadata leaves.
"""
import threading

from harpe import ContextBudgetEngine, ContextItem, TokenBudget
from harpe.two_pass import TwoPassBuild


def _report(metadata):
    items = [
        ContextItem(id="row", content="short description", token_count=5, metadata=metadata),
        ContextItem(id="rule", content="always answer in English", token_count=6, pinned=True),
    ]
    return ContextBudgetEngine().synthesize(items, TokenBudget(total_budget=1000))


def _bodies():
    return {"row": ContextItem(id="row", content="the full body " * 10, token_count=40)}


def test_control_plain_metadata_builds_and_expands():
    build = TwoPassBuild(_report({"tags": ["a"]}), bodies=_bodies(), budget_tokens=500)
    assert build.request("row") is True
    assert [it.id for it in build.items()] == ["row", "rule"]
    assert build.used_tokens == 46


def test_live_object_in_item_metadata_does_not_break_the_build():
    report = _report({"source_lock": threading.Lock()})
    build = TwoPassBuild(report, bodies=_bodies(), budget_tokens=500)
    assert build.request("row") is True
    assert [it.id for it in build.items()] == ["row", "rule"]


def test_the_build_still_owns_what_its_receipt_accounts_for():
    """The property the copies were added for; must keep holding under any replacement."""
    report = _report({"tags": ["a"]})
    bodies = _bodies()
    build = TwoPassBuild(report, bodies=bodies, budget_tokens=500)
    bodies["row"].content = "edited after the build"
    bodies["row"].token_count = 400
    report.included_items[0].content = "edited after the build"
    assert build.request("row") is True
    assert build.used_tokens == 46
    assert [it.content for it in build.items()] == ["the full body " * 10, "always answer in English"]
