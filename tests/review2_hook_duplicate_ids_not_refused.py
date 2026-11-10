"""severity: medium
finding: synthesize refuses duplicate item ids only before the pre_prune hooks, so a hook that re-adds an item a replayed report already carries yields a report with two items of one id

ContextBudgetEngine._normalise_items (src/harpe/engine.py:342-364) refuses duplicate ids: "a report keyed by id
would still be ambiguous for the caller. Refuse them rather than guess" (engine.py:358-363). synthesize
(engine.py:562) runs that check on the caller's items, then hands the items to the ``pre_prune`` hooks
(engine.py:568-570). The round-1 fix re-runs the sensitivity check and the token count after the hooks "so the
items selected are the items checked" (engine.py:571-576), but not the id check, so the id rule holds only for the
items as they were before the hooks.

The common path that trips it is a replay: a hook injects a standing item (a reminder, a policy row), turn 1's
report includes it, and turn 2 is synthesized over turn 1's included items plus the new turn. The caller's list has
the injected item once, the hook adds it again, and turn 2's report includes two items with the id ``reminder``,
each taking budget. The same list given with the duplicate up front raises ``ValueError``. A ``TwoPassBuild`` over
such a report keys its rows by id (see review2_two_pass_duplicate_ids_receipt_misdescribes.py).

The test accepts either fix: synthesize refuses the duplicate the hook made (``ValueError``/``TypeError``), or returns a report
whose ids are unique.
"""
from __future__ import annotations

from typing import List, Sequence

import pytest

from harpe import ContextBudgetEngine, ContextItem, TokenBudget


class InjectReminder:
    """A pre_prune hook that adds a standing reminder to every synthesis."""

    def pre_prune(self, items: Sequence[ContextItem]) -> None:
        items.append(ContextItem(id="reminder", content="Cite your sources.", token_count=5))  # type: ignore[attr-defined]

    def post_prune(self, report) -> None:
        pass


def _budget() -> TokenBudget:
    return TokenBudget(total_budget=1000)


def _report_ids(report) -> List[str]:
    return [it.id for it in report.included_items + report.pruned_items + report.withheld_items]


def test_control_duplicate_ids_from_the_caller_are_refused():
    items = [ContextItem(id="reminder", content="Cite your sources.", token_count=5),
             ContextItem(id="reminder", content="Cite your sources.", token_count=5)]
    with pytest.raises(ValueError):
        ContextBudgetEngine().synthesize(items, _budget())


def test_control_the_hook_adds_its_reminder_once_on_a_fresh_turn():
    report = ContextBudgetEngine(hooks=[InjectReminder()]).synthesize(
        [ContextItem(id="q1", content="What failed in last night's deploy?", token_count=9)], _budget())
    assert [it.id for it in report.included_items] == ["q1", "reminder"]


def test_a_replayed_turn_does_not_come_back_with_two_items_of_one_id():
    engine = ContextBudgetEngine(hooks=[InjectReminder()])
    turn1 = engine.synthesize([ContextItem(id="q1", content="What failed in last night's deploy?", token_count=9)],
                              _budget())
    carried = list(turn1.included_items) + [ContextItem(id="q2", content="And who rolled it back?", token_count=7)]
    try:
        turn2 = engine.synthesize(carried, _budget())
    except (ValueError, TypeError):
        return  # refusing the duplicate the hook made is a correct fix
    ids = _report_ids(turn2)
    assert len(ids) == len(set(ids)), f"turn 2's report carries duplicate ids: {ids}"
