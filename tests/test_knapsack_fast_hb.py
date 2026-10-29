"""H-B: the exact 0/1 knapsack DP must stay bit-for-bit faithful to the textbook table while running materially
faster. The reference solver below is the plain two-dimensional table with the standard backtrace (take item
``i`` at cell ``c`` iff the row changed there), which is exactly the strict-improvement rule the engine's
one-row implementation records in its keep rows."""

import random
import time

import pytest

from harpe import ContextBudgetEngine, ContextItem, PrunePolicy, TokenBudget

solve = ContextBudgetEngine._solve_knapsack_dp


def reference_knapsack(weights, utilities, capacity):
    """Textbook 2-D 0/1 knapsack: ``table[i][c]`` is the best utility using items ``< i`` within weight ``c``.

    Returns the set of selected indices. Ties are resolved exactly as the engine does: an item is taken only
    when it strictly improves the cell, so with equal utilities the earlier items win."""
    n = len(weights)
    table = [[0.0] * (capacity + 1)]
    for i in range(n):
        prev = table[-1]
        row = list(prev)
        w, u = weights[i], utilities[i]
        for c in range(w, capacity + 1):
            cand = prev[c - w] + u
            if cand > row[c]:
                row[c] = cand
        table.append(row)
    take = set()
    c = capacity
    for i in range(n - 1, -1, -1):
        if table[i + 1][c] != table[i][c]:
            take.add(i)
            c -= weights[i]
    return take


def make_item(n, tokens, **kw):
    kw.setdefault("priority", 50)
    return ContextItem(id=f"i{n}", content=f"item {n}", token_count=tokens, created_at=float(n), **kw)


def random_instance(rng, n, capacity):
    """Weights include items larger than the capacity; utilities are drawn from a small pool so ties are
    frequent (roughly a third of the items share a utility with another one)."""
    weights = []
    for _ in range(n):
        roll = rng.random()
        if roll < 0.08:
            weights.append(rng.randint(capacity + 1, capacity + 50))  # can never fit
        elif roll < 0.2:
            weights.append(rng.choice([3, 5, 7, 11]))  # shared weights -> more ties
        else:
            weights.append(rng.randint(1, max(2, capacity // 3)))
    pool = [float(rng.choice([1, 2, 3, 5, 8])) for _ in range(4)]
    utilities = [rng.choice(pool) if rng.random() < 0.35 else rng.random() * 10.0 for _ in range(n)]
    return weights, utilities


@pytest.mark.parametrize("seed", range(50))
def test_fast_dp_matches_reference_selection_on_random_instances(seed):
    rng = random.Random(1000 + seed)
    n = rng.randint(1, 60)
    capacity = rng.randint(1, 400)
    weights, utilities = random_instance(rng, n, capacity)
    items = [make_item(k, weights[k]) for k in range(n)]
    chosen, rejected = solve(items, utilities, capacity)
    expected = reference_knapsack(weights, utilities, capacity)
    assert [int(it.id[1:]) for it in chosen] == sorted(expected)  # same set, input order preserved
    assert [int(it.id[1:]) for it in rejected] == sorted(set(range(n)) - expected)
    assert sum(it.token_count for it in chosen) <= capacity


@pytest.mark.parametrize("seed", range(50))
def test_synthesize_knapsack_matches_reference_with_pinned_items(seed):
    """Through ``synthesize``: pinned items are always kept and their tokens come off the top; the rest is the
    exact DP over the engine's own utilities on the remaining capacity."""
    rng = random.Random(5000 + seed)
    n = rng.randint(1, 40)
    capacity = rng.randint(20, 300)
    weights, _ = random_instance(rng, n, capacity)
    items = []
    for k in range(n):
        pinned = rng.random() < 0.15 and weights[k] <= capacity // 4
        items.append(
            make_item(
                k,
                weights[k],
                pinned=pinned,
                priority=rng.randint(1, 100),
                metadata={"relevance": rng.choice([0.25, 0.5, 0.75, rng.random()])},
            )
        )
    engine = ContextBudgetEngine()
    budget = TokenBudget(total_budget=capacity)
    report = engine.synthesize(items, budget, PrunePolicy.KNAPSACK_DP)

    pinned = [it for it in items if it.pinned]
    unpinned = [it for it in items if not it.pinned]
    remaining = capacity - sum(it.token_count for it in pinned)
    reference_time = max(it.created_at for it in items)
    expected_ids = {it.id for it in pinned}
    if remaining > 0:
        utilities = [ContextBudgetEngine._compute_item_utility(it, now=reference_time) for it in unpinned]
        take = reference_knapsack([it.token_count for it in unpinned], utilities, remaining)
        expected_ids |= {unpinned[k].id for k in take}
    assert {it.id for it in report.included_items} == expected_ids
    assert [it.id for it in report.included_items] == [it.id for it in items if it.id in expected_ids]


def test_items_heavier_than_the_capacity_are_rejected_and_leave_the_rest_unchanged():
    items = [make_item(0, 6), make_item(1, 11), make_item(2, 5), make_item(3, 5), make_item(4, 50)]
    chosen, rejected = solve(items, [10.0, 100.0, 9.0, 9.0, 100.0], 10)
    assert [it.id for it in chosen] == ["i2", "i3"]
    assert [it.id for it in rejected] == ["i0", "i1", "i4"]
    chosen, rejected = solve([make_item(0, 11)], [1.0], 10)
    assert chosen == [] and [it.id for it in rejected] == ["i0"]


def test_common_divisor_reduction_is_lossless():
    # Every weight is a multiple of 50, so the DP runs over capacity // 50 cells and must pick the same set
    # as the reference does on the raw weights, ties included.
    rng = random.Random(42)
    for _ in range(20):
        n = rng.randint(1, 30)
        weights = [50 * rng.randint(1, 8) for _ in range(n)]
        utilities = [float(rng.choice([1, 2, 3, 5])) for _ in range(n)]
        capacity = rng.randint(1, 1200)
        items = [make_item(k, weights[k]) for k in range(n)]
        chosen, _ = solve(items, utilities, capacity)
        assert [int(it.id[1:]) for it in chosen] == sorted(reference_knapsack(weights, utilities, capacity))


def test_synthesize_200_items_at_8000_tokens_is_fast():
    """200 items with 20-400 tokens each against an 8,000-token ceiling: n*(capacity+1) = 1.6M cells, well
    inside the exact path. The bound is relative, not a wall-clock figure: the fast DP must take at most 60 %
    of the textbook double loop on the same instance in the same process (0.32 on the development host,
    0.39 against the pre-H-B one-row loop), because an absolute millisecond bound depends on the runner.
    The release run for v0.1.0-alpha.11 failed on a 26 ms bound that a GitHub runner on CPython 3.12 took 50 ms
    to meet."""
    rng = random.Random(1234)
    items = [
        ContextItem(
            id=f"i{k}",
            content="x",
            token_count=rng.randint(20, 400),
            priority=rng.randint(1, 100),
            created_at=1_700_000_000.0 + k,
            metadata={"relevance": rng.random()},
        )
        for k in range(200)
    ]
    engine = ContextBudgetEngine()
    budget = TokenBudget(total_budget=8000)
    assert budget.available_context == 8000
    report = engine.synthesize(items, budget, PrunePolicy.KNAPSACK_DP)
    assert report.consumed_tokens <= 8000
    assert 8000 - report.consumed_tokens < max(it.token_count for it in items)

    weights = [it.token_count for it in items]
    utilities = [rng.random() * 100 for _ in items]
    assert ContextBudgetEngine._exact_knapsack(weights, utilities, 8000) == reference_knapsack(weights, utilities, 8000)

    def best_of(fn, runs=5):
        fn()  # warm-up
        return min(_timed(fn) for _ in range(runs))

    slow = best_of(lambda: reference_knapsack(weights, utilities, 8000))
    fast = best_of(lambda: ContextBudgetEngine._exact_knapsack(weights, utilities, 8000))
    assert fast <= 0.6 * slow, f"fast DP {fast * 1000:.1f} ms vs textbook {slow * 1000:.1f} ms (ratio {fast / slow:.2f})"


def _timed(fn):
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start
