"""severity: high
finding: a pre_prune hook that marks an item confidential cannot withhold it: synthesize() checks sensitivity before the hooks run, so the item is sent to the remote destination

ContextBudgetEngine.synthesize (src/harpe/engine.py:471) withholds ``sensitivity == "confidential"`` items for
``destination="remote"`` at engine.py:522-523, then copies the items for the hooks (engine.py:524 and
_count_uncounted, engine.py:326-333: "Hooks may change items in place, so with hooks every item is copied") and
only then calls ``PruningHook.pre_prune`` (engine.py:526). protocols.py:22-23 documents pre_prune as "Invoked
prior to any item filtering or pruning", and synthesize's docstring promises that remote "withholds every item
marked confidential, pinned or not, before any budgeting, so it cannot take a budget slot or appear in the
output". The check runs on the items as they were before the hooks and selection runs on the items as the hooks
left them: an item that a classification hook marks confidential is selected into ``included_items`` (and so
into ``to_messages()``) for the remote destination, still marked confidential, and ``withheld_items`` omits it.
"""
from __future__ import annotations

from harpe import ContextBudgetEngine, ContextItem, TokenBudget


class MarkCredentialsConfidential:
    """A classification hook: an item that carries a credential marker is confidential."""

    def pre_prune(self, items):
        for it in items:
            if "API_KEY=" in it.content:
                it.sensitivity = "confidential"

    def post_prune(self, report):
        pass


def _items():
    return [
        ContextItem(id="policy", content="Answer briefly.", pinned=True, token_count=5),
        ContextItem(id="note", content="The deploy finished at 10:02.", token_count=8),
        ContextItem(id="env", content="env dump: API_KEY=sk-test-0000", token_count=8),
    ]


def _budget() -> TokenBudget:
    return TokenBudget(total_budget=1000, reserved_completion=0, reserved_tools=0, safety_margin=0)


def _engine() -> ContextBudgetEngine:
    return ContextBudgetEngine(hooks=[MarkCredentialsConfidential()])


def test_control_an_item_marked_confidential_up_front_is_withheld_for_remote():
    items = _items()
    items[2] = items[2].model_copy(update={"sensitivity": "confidential"})
    report = _engine().synthesize(items, _budget(), destination="remote")
    assert [it.id for it in report.withheld_items] == ["env"]
    assert [it.id for it in report.included_items] == ["policy", "note"]


def test_control_the_hook_runs_and_local_keeps_everything():
    report = _engine().synthesize(_items(), _budget(), destination="local")
    included = {it.id: it for it in report.included_items}
    assert set(included) == {"policy", "note", "env"}
    assert included["env"].sensitivity == "confidential"  # the hook classified the copy it was given


def test_an_item_a_hook_marks_confidential_is_not_sent_to_a_remote_destination():
    report = _engine().synthesize(_items(), _budget(), destination="remote")
    leaked = [it.id for it in report.included_items if it.sensitivity == "confidential"]
    assert leaked == [], f"confidential items in a remote report's included_items: {leaked}"
    assert all("API_KEY=" not in m["content"] for m in report.to_messages())
