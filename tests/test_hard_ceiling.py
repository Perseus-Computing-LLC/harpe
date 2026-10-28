"""The hard token ceiling: selection is tracked by object identity, duplicate ids are refused, and every policy's
output fits the budget it was given."""

import pytest

from harpe import ContextBudgetEngine, ContextItem, PrunePolicy, TokenBudget


def test_duplicate_ids_are_refused_instead_of_emitting_both():
    # Before the fix two items sharing an id, 100 and 1000 tokens, under a 500-token budget came back with
    # consumed_tokens=1100 and nothing pruned: the final list was rebuilt by id membership.
    small = ContextItem(id="same", content="small", token_count=100, priority=10)
    large = ContextItem(id="same", content="large", token_count=1000, priority=20)
    with pytest.raises(ValueError, match="duplicate ContextItem ids"):
        ContextBudgetEngine().synthesize([small, large], TokenBudget(total_budget=500))


@pytest.mark.parametrize("policy", list(PrunePolicy))
def test_every_policy_stays_under_the_ceiling(policy):
    items = [
        ContextItem(id=f"i{n}", content=f"item {n}", token_count=100 + 37 * (n % 5), priority=5 + n, created_at=float(n))
        for n in range(30)
    ]
    budget = TokenBudget(total_budget=900)
    report = ContextBudgetEngine().synthesize(items, budget, policy=policy)
    assert report.consumed_tokens <= budget.available_context
    assert report.consumed_tokens == sum(it.token_count for it in report.included_items)
    assert len(report.included_items) + len(report.pruned_items) == len(items)
    # Each input object lands in exactly one list and the lists keep input order.
    seen = [id(it) for it in report.included_items] + [id(it) for it in report.pruned_items]
    assert len(set(seen)) == len(items)
    order = {id(it): n for n, it in enumerate(items)}
    assert [order[id(it)] for it in report.included_items] == sorted(order[id(it)] for it in report.included_items)
