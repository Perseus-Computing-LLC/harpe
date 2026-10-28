"""The knapsack selection behind KNAPSACK_DP and DUO_ATTENTION: exact on raw token weights when affordable, density
greedy with slack fill otherwise, never a quantised budget."""

import pytest

from harpe import ContextBudgetEngine, ContextItem, PrunePolicy, TokenBudget

solve = ContextBudgetEngine._solve_knapsack_dp


def item(n, tokens, **kw):
    return ContextItem(id=f"i{n}", content=f"item {n}", token_count=tokens, priority=50, created_at=float(n), **kw)


def test_exact_dp_finds_the_best_combination_not_the_greedy_one():
    # Density greedy takes the 6-token item (utility 10, density 1.67) and then nothing else fits; the exact
    # answer is the two 5-token items (utility 9 each).
    items = [item(0, 6), item(1, 5), item(2, 5)]
    chosen, rejected = solve(items, [10.0, 9.0, 9.0], 10)
    assert [it.id for it in chosen] == ["i1", "i2"]
    assert [it.id for it in rejected] == ["i0"]


def test_exact_dp_uses_each_item_once_and_keeps_input_order():
    items = [item(n, 3) for n in range(5)]
    chosen, rejected = solve(items, [1.0, 5.0, 2.0, 5.0, 1.0], 6)
    assert [it.id for it in chosen] == ["i1", "i3"]
    assert [it.id for it in rejected] == ["i0", "i2", "i4"]
    assert sum(it.token_count for it in chosen) <= 6


def test_small_budget_runs_the_exact_path_on_raw_weights():
    # Before the fix, capacity 2000 with 1000 bins made every item cost at least 2 tokens of budget.
    items = [item(n, 1) for n in range(1200)]
    chosen, rejected = solve(items, [1.0] * 1200, 2000)
    assert len(chosen) == 1200 and rejected == []


def test_large_budget_falls_back_to_greedy_and_includes_everything_that_fits():
    # 1,500 one-token items at a 200k budget: n*(capacity+1) is 300M cells, so this is the greedy path.
    # Under the old 1000-bin quantisation only 1,000 of them survived.
    items = [item(n, 1) for n in range(1500)]
    chosen, rejected = solve(items, [1.0] * 1500, 200_000)
    assert len(chosen) == 1500 and rejected == []


def test_greedy_slack_fill_recovers_items_density_skipped():
    # Density order: i0 (4.0/tok), i1 (1.0/tok), i2 (0.5/tok). Greedy takes i0 (2 tokens) and i1 (6 tokens) and then
    # cannot fit i2 (10 tokens) in the 2 remaining; slack fill in utility order adds i3 (2 tokens) which fits.
    items = [item(0, 2), item(1, 6), item(2, 10), item(3, 2)]
    chosen, _ = solve(items, [8.0, 6.0, 5.0, 0.5], 10, max_cells=1)
    assert [it.id for it in chosen] == ["i0", "i1", "i3"]


@pytest.mark.parametrize("max_cells", [5_000_000, 1])
def test_unused_capacity_is_below_the_largest_item_when_demand_exceeds_capacity(max_cells):
    items = [item(n, 20 + (n * 7) % 50) for n in range(60)]
    capacity = 500
    chosen, _ = solve(items, [float(1 + n % 7) for n in range(60)], capacity, max_cells=max_cells)
    consumed = sum(it.token_count for it in chosen)
    assert consumed <= capacity
    assert consumed >= capacity - max(it.token_count for it in items)


@pytest.mark.parametrize("policy", [PrunePolicy.KNAPSACK_DP, PrunePolicy.DUO_ATTENTION])
def test_policies_fill_the_budget_within_one_item_of_capacity(policy):
    items = [item(n, 30 + (n * 11) % 40) for n in range(40)]
    budget = TokenBudget(total_budget=700)
    report = ContextBudgetEngine().synthesize(items, budget, policy=policy)
    capacity = budget.available_context
    assert report.consumed_tokens <= capacity
    assert report.consumed_tokens >= capacity - max(it.token_count for it in items)
    assert len(report.included_items) + len(report.pruned_items) == len(items)


def test_duo_attention_keeps_four_sinks_and_four_local_turns_first():
    items = [item(n, 10) for n in range(12)]
    report = ContextBudgetEngine().synthesize(items, TokenBudget(total_budget=85), policy=PrunePolicy.DUO_ATTENTION)
    ids = {it.id for it in report.included_items}
    assert {"i0", "i1", "i2", "i3", "i8", "i9", "i10", "i11"} <= ids
    assert report.consumed_tokens == 80


def test_knapsack_policy_at_a_200k_budget_includes_1500_one_token_items():
    items = [item(n, 1) for n in range(1500)]
    report = ContextBudgetEngine().synthesize(items, TokenBudget(total_budget=200_000), policy=PrunePolicy.KNAPSACK_DP)
    assert len(report.included_items) == 1500
    assert report.pruned_items == []
    assert report.consumed_tokens == 1500
