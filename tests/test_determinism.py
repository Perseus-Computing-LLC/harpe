"""KNAPSACK_DP and DUO_ATTENTION score recency from the inputs, not the wall clock, so a replay is identical."""

import time
from unittest import mock

import pytest

from harpe import ContextBudgetEngine, ContextItem, PrunePolicy, TokenBudget


def _items():
    base = 1_700_000_000.0
    return [
        ContextItem(
            id=f"i{n}",
            content=f"turn {n}",
            token_count=40 + (n * 13) % 30,
            priority=20 + (n * 7) % 60,
            created_at=base + n * 3600.0,
            metadata={"attention_density": 0.3 + (n % 4) * 0.2},
        )
        for n in range(16)
    ]


@pytest.mark.parametrize("policy", [PrunePolicy.KNAPSACK_DP, PrunePolicy.DUO_ATTENTION])
def test_same_inputs_give_the_same_output_whatever_the_clock_says(policy):
    engine = ContextBudgetEngine()
    budget = TokenBudget(total_budget=400)
    first = engine.synthesize(_items(), budget, policy=policy)
    # Pretend 30 days have passed; before the fix a 3-hour gap already moved items near the margin.
    with mock.patch("time.time", return_value=time.time() + 30 * 86400):
        second = engine.synthesize(_items(), budget, policy=policy)
    assert [it.id for it in first.included_items] == [it.id for it in second.included_items]
    assert first.model_dump() == second.model_dump()
    assert first.reference_time == max(it.created_at for it in _items())


def test_explicit_now_is_used_and_reported():
    engine = ContextBudgetEngine()
    items = _items()
    report = engine.synthesize(items, TokenBudget(total_budget=400), policy=PrunePolicy.KNAPSACK_DP, now=1_800_000_000.0)
    assert report.reference_time == 1_800_000_000.0
    assert report.consumed_tokens <= 400


def test_reference_time_is_set_on_the_pinned_overflow_path_and_none_without_items():
    pinned = ContextItem(id="p", content="pinned", token_count=900, pinned=True, created_at=123.0)
    report = ContextBudgetEngine().synthesize([pinned], TokenBudget(total_budget=500))
    assert report.reference_time == 123.0
    assert ContextBudgetEngine().synthesize([], TokenBudget(total_budget=500)).reference_time is None


def test_item_utility_without_a_reference_treats_the_item_as_current():
    item = ContextItem(id="a", content="a", token_count=1, priority=50, created_at=1.0)
    assert ContextBudgetEngine._compute_item_utility(item) == ContextBudgetEngine._compute_item_utility(item, now=1.0)
    assert ContextBudgetEngine._compute_item_utility(item, now=1.0 + 36_000) < ContextBudgetEngine._compute_item_utility(item)
