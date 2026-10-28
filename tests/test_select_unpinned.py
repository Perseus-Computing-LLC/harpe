"""_select_unpinned is the single place a policy is applied; an unknown policy is an error, not a silent fallback."""

import pytest

from harpe import ContextBudgetEngine, ContextItem, PrunePolicy, TokenBudget


def _items(n=10):
    return [ContextItem(id=f"i{k}", content=f"item {k}", token_count=50 + k, priority=10 + k, created_at=float(k)) for k in range(n)]


def test_unknown_policy_raises_instead_of_falling_back_to_lowest_priority():
    engine = ContextBudgetEngine()
    with pytest.raises(ValueError, match="unknown prune policy"):
        engine.synthesize(_items(), TokenBudget(total_budget=300), policy="not_a_policy")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown prune policy"):
        engine._select_unpinned("not_a_policy", _items(), 300)  # type: ignore[arg-type]


@pytest.mark.parametrize("policy", list(PrunePolicy))
def test_select_unpinned_partitions_the_input_and_respects_capacity(policy):
    items = _items()
    included, pruned = ContextBudgetEngine()._select_unpinned(policy, items, 230, reference_time=9.0)
    assert sum(it.token_count for it in included) <= 230
    assert sorted(it.id for it in included + pruned) == sorted(it.id for it in items)
    assert not ({id(it) for it in included} & {id(it) for it in pruned})


def test_select_unpinned_matches_synthesize_for_each_policy():
    engine = ContextBudgetEngine()
    for policy in PrunePolicy:
        items = _items()
        report = engine.synthesize(items, TokenBudget(total_budget=230), policy=policy)
        included, _ = engine._select_unpinned(policy, items, 230, reference_time=report.reference_time)
        assert sorted(it.id for it in included) == sorted(it.id for it in report.included_items), policy
