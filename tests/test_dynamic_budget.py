"""Dynamic budget expansion never shrinks the caller's budget, and the report says which ceiling was used."""

from harpe import ContextBudgetEngine, ContextItem, PrunePolicy, QueryIntent, TokenBudget
from harpe.routing import resolve_dynamic_budget


def test_budget_above_the_old_absolute_cap_grows_instead_of_shrinking():
    # Before the fix total_budget=8000 with aggregation intent came back as 4000.
    base = TokenBudget(total_budget=8000, reserved_completion=500)
    expanded = resolve_dynamic_budget(base, QueryIntent.MULTI_SESSION_AGGREGATION)
    assert expanded.total_budget == 16000
    assert expanded.reserved_completion == 500
    assert expanded.total_budget >= base.total_budget


def test_factor_is_clamped_to_max_factor_and_never_below_one():
    base = TokenBudget(total_budget=10_000)
    assert resolve_dynamic_budget(base, "aggregation", aggregation_factor=5.0, max_factor=1.5).total_budget == 15_000
    assert resolve_dynamic_budget(base, "aggregation", aggregation_factor=0.25).total_budget == 10_000
    assert resolve_dynamic_budget(base, "aggregation", aggregation_factor=3.0, max_factor=0.5).total_budget == 10_000
    assert resolve_dynamic_budget(base, QueryIntent.TEMPORAL, aggregation_factor=5.0) is base


def test_report_carries_the_effective_budget_on_every_path():
    engine = ContextBudgetEngine()
    items = [ContextItem(id=f"i{n}", content=f"session fact {n}", token_count=1000, priority=n + 1) for n in range(12)]
    budget = TokenBudget(total_budget=6000, reserved_completion=1000)

    plain = engine.synthesize(items, budget)
    assert plain.effective_budget == 5000
    assert plain.consumed_tokens <= plain.effective_budget

    widened = engine.synthesize(items, budget, intent=QueryIntent.MULTI_SESSION_AGGREGATION, dynamic_budget=True)
    assert widened.dynamic_budget_expanded is True
    assert widened.effective_budget == 11_000
    assert widened.consumed_tokens == 11_000
    assert len(widened.included_items) == 11

    # The pinned-exceeds-capacity early return reports the same ceiling.
    pinned = ContextItem(id="pin", content="pinned", token_count=9000, pinned=True)
    over = engine.synthesize([pinned, *items], budget, policy=PrunePolicy.FIFO)
    assert over.effective_budget == 5000
    assert over.included_items == [pinned]
