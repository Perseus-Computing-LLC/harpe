"""severity: medium
finding: ContextItem validation writes the generated id back into the caller's dict, so a later copy of that dict replays the old item's id and synthesize() refuses it as a duplicate

ContextItem._resolve_item_id_alias (src/harpe/models.py:109-123) is a mode="before" validator that assigns into
the input mapping itself: a dict without an id gets ``data["id"] = data["item_id"] = uuid4()``, and a dict with an
id gets ``data["item_id"]``. ContextBudgetEngine._normalise_items validates every dict item that way
(src/harpe/engine.py:310), so ``synthesize`` rewrites the caller's rows. The generated id then lives in the
caller's data: a row built later as a copy of an earlier one (``dict(row, content=...)``) carries that id, and the
engine refuses the call with "duplicate ContextItem ids" for an id the caller never set (engine.py:317-323). An
id-less item is supposed to get its own generated id; the caller's input should be left as it was.
"""
from __future__ import annotations

import copy

import pytest

from harpe import ContextBudgetEngine, ContextItem, TokenBudget


def _budget() -> TokenBudget:
    return TokenBudget(total_budget=1000, reserved_completion=0, reserved_tools=0, safety_margin=0)


def _rows():
    return [{"content": "alpha: the deploy finished.", "token_count": 6},
            {"content": "beta: the cache was warmed.", "token_count": 6}]


def test_control_fresh_id_less_rows_get_distinct_ids_and_all_fit():
    report = ContextBudgetEngine().synthesize(_rows() + [{"content": "gamma: tests passed.", "token_count": 4}],
                                              _budget())
    ids = [it.id for it in report.included_items]
    assert len(ids) == 3 and len(set(ids)) == 3 and all(ids)


def test_control_explicit_duplicate_ids_are_refused():
    rows = [{"id": "a", "content": "one", "token_count": 1}, {"id": "a", "content": "two", "token_count": 1}]
    with pytest.raises(ValueError):
        ContextBudgetEngine().synthesize(rows, _budget())


def test_synthesize_leaves_the_callers_row_dicts_unchanged():
    rows = _rows()
    before = copy.deepcopy(rows)
    ContextBudgetEngine().synthesize(rows, _budget())
    assert rows == before, f"synthesize wrote into the caller's rows: {rows}"


def test_a_copy_of_an_earlier_row_is_a_new_item_not_a_replay_of_its_id():
    engine = ContextBudgetEngine()
    rows = _rows()
    engine.synthesize(rows, _budget())
    rows.append(dict(rows[0], content="alpha: the deploy finished, second run."))
    try:
        report = engine.synthesize(rows, _budget())
    except ValueError as exc:
        raise AssertionError(f"the copied row replayed an id the caller never set: {exc}") from exc
    ids = [it.id for it in report.included_items]
    assert len(ids) == 3 and len(set(ids)) == 3
