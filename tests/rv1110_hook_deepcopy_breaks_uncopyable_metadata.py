"""severity: medium
finding: regression from the round-2 fix (a02a145): with any hook registered, synthesize raises TypeError when an item's metadata holds an object deepcopy cannot copy (a lock, an open handle, a client); it worked before the fix and still works without hooks

G2 was fixed by `model_copy(deep=True)` on every item whenever hooks are set (src/harpe/engine.py:375-381, and the
withheld items at :579). `ContextItem.metadata` is `Dict[str, Any]`, "Arbitrary item metadata" (models.py:99), and
callers do carry live objects there (a tool handle, a source client, a lock guarding the record). `copy.deepcopy`
cannot copy those and raises `TypeError: cannot pickle '_thread.lock' object` from inside `synthesize`. At 044ea15
the copy was shallow and the same call returned a report. Registering a no-op hook is enough to trigger it, so
adding observability to a working pipeline now breaks it.

The fix for G2 needs the hook's copy to own its metadata containers; it does not need to clone opaque leaf objects.
"""
import threading

import pytest

from harpe import ContextBudgetEngine, ContextItem, TokenBudget


class _Observer:
    def __init__(self):
        self.seen = 0

    def pre_prune(self, items):
        self.seen = len(items)

    def post_prune(self, report):
        pass


def _items():
    return [
        ContextItem(id="doc", content="alpha beta gamma", metadata={"source_lock": threading.Lock(), "tags": ["a"]}),
        ContextItem(id="note", content="delta epsilon"),
    ]


def test_control_no_hooks_accepts_the_metadata():
    report = ContextBudgetEngine().synthesize(_items(), TokenBudget(total_budget=1000), destination="local")
    assert [it.id for it in report.included_items] == ["doc", "note"]


def test_control_hook_with_plain_metadata_works():
    hook = _Observer()
    items = [ContextItem(id="doc", content="alpha beta gamma", metadata={"tags": ["a"]})]
    report = ContextBudgetEngine(hooks=[hook]).synthesize(items, TokenBudget(total_budget=1000), destination="local")
    assert hook.seen == 1 and [it.id for it in report.included_items] == ["doc"]


def test_a_hook_does_not_make_synthesize_reject_live_objects_in_metadata():
    hook = _Observer()
    engine = ContextBudgetEngine(hooks=[hook])
    report = engine.synthesize(_items(), TokenBudget(total_budget=1000), destination="local")
    assert hook.seen == 2
    assert [it.id for it in report.included_items] == ["doc", "note"]


def test_withheld_items_with_live_objects_do_not_break_a_remote_build():
    items = _items() + [
        ContextItem(id="secret", content="zeta", sensitivity="confidential", metadata={"handle": threading.Lock()})
    ]
    report = ContextBudgetEngine(hooks=[_Observer()]).synthesize(items, TokenBudget(total_budget=1000))
    assert [it.id for it in report.withheld_items] == ["secret"]


def test_g2_stays_fixed_a_hook_write_does_not_reach_the_caller():
    """The property the deep copy was added for; must keep holding under any replacement."""

    class _Scorer(_Observer):
        def pre_prune(self, items):
            for it in items:
                it.metadata.setdefault("tags", []).append("scored")

    items = [ContextItem(id="doc", content="alpha beta gamma", metadata={"tags": ["a"]})]
    ContextBudgetEngine(hooks=[_Scorer()]).synthesize(items, TokenBudget(total_budget=1000), destination="local")
    assert items[0].metadata == {"tags": ["a"]}
