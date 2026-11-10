"""severity: low
finding: OpenLoopTracker.to_context_items spreads the loop's caller metadata over loop_id and status, so a follow-up registered with an earlier item's metadata names the earlier loop and reports its stale status

to_context_items (src/harpe/open_loops.py:104-120, line 117) builds each item's metadata as
{"loop_id": loop.id, "status": loop.status.value, **loop.metadata}: the free metadata given to register()
(open_loops.py:50-65) is spread last and wins over the two keys that identify the loop and its lifecycle state.
Carrying an item's metadata forward to a follow-up loop (keeping its tags) replays the earlier loop's
``loop_id``: the follow-up's item names the earlier loop while its own id and content name the follow-up, and a
caller that resolves through ``item.metadata["loop_id"]`` writes to the earlier (already resolved) loop and
leaves the follow-up open. The same spread lets any caller key named ``status`` replace the real status.
"""
from __future__ import annotations

from harpe.open_loops import OpenLoopStatus, OpenLoopTracker


def test_control_item_metadata_names_the_loop_and_its_status():
    tracker = OpenLoopTracker()
    loop = tracker.register("Ship the fix", metadata={"owner": "ops"})
    tracker.update_status(loop.id, OpenLoopStatus.IN_PROGRESS)
    (item,) = tracker.to_context_items()
    assert item.id == f"open_loop_{loop.id}"
    assert (item.metadata["loop_id"], item.metadata["status"], item.metadata["owner"]) == (loop.id, "in_progress",
                                                                                           "ops")


def test_a_follow_up_registered_with_an_earlier_items_metadata_names_itself():
    tracker = OpenLoopTracker()
    first = tracker.register("Ship the fix", metadata={"owner": "ops"})
    tracker.update_status(first.id, OpenLoopStatus.IN_PROGRESS)
    (first_item,) = tracker.to_context_items()
    tracker.resolve(first.id, "shipped")
    follow = tracker.register("Verify the fix in production", metadata=dict(first_item.metadata))
    (item,) = tracker.to_context_items()
    assert item.id == f"open_loop_{follow.id}"
    assert item.metadata["owner"] == "ops"
    assert item.metadata["loop_id"] == follow.id, "the follow-up's item names the earlier loop"
    assert item.metadata["status"] == "open"
