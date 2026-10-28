"""Pinned, priority-0 and invariant-dimension items are never pruned. When they alone exceed the ceiling the report
says so, and strict callers get an exception."""

import pytest

from harpe import BudgetOverflowError, CognitiveDimension, ContextBudgetEngine, ContextItem, PrunePolicy, TokenBudget


def test_pinned_overflow_is_flagged_not_hidden():
    # Before the fix: consumed_tokens=99,999, budget_remaining=0 and nothing else to tell the caller.
    pinned = ContextItem(id="sys", content="system prompt", token_count=99_999, priority=0)
    other = ContextItem(id="x", content="x", token_count=10, priority=50)
    report = ContextBudgetEngine().synthesize([pinned, other], TokenBudget(total_budget=8000))
    assert report.over_budget is True
    assert report.overflow_tokens == 99_999 - 8000
    assert report.consumed_tokens == 99_999
    assert report.effective_budget == 8000
    assert report.budget_remaining == 0
    assert report.included_items == [pinned]
    assert report.pruned_items == [other]


def test_invariant_dimension_items_count_as_pinned_for_the_overflow():
    inv = ContextItem(id="inv", content="policy", token_count=600, priority=50, dimension=CognitiveDimension.INVARIANT)
    report = ContextBudgetEngine().synthesize([inv], TokenBudget(total_budget=500))
    assert report.over_budget is True and report.overflow_tokens == 100
    assert report.included_items == [inv]


def test_strict_raises_with_the_report_attached():
    pinned = ContextItem(id="sys", content="system prompt", token_count=700, pinned=True)
    with pytest.raises(BudgetOverflowError, match="200 over") as info:
        ContextBudgetEngine().synthesize([pinned], TokenBudget(total_budget=500), strict=True)
    assert info.value.report.overflow_tokens == 200
    assert isinstance(info.value, ValueError)


def test_exactly_full_is_not_over_budget_and_strict_returns():
    pinned = ContextItem(id="sys", content="system prompt", token_count=500, pinned=True)
    report = ContextBudgetEngine().synthesize([pinned], TokenBudget(total_budget=500), strict=True)
    assert report.over_budget is False and report.overflow_tokens == 0


@pytest.mark.parametrize("policy", list(PrunePolicy))
def test_normal_path_reports_no_overflow(policy):
    items = [ContextItem(id=f"i{n}", content=f"item {n}", token_count=100, priority=10 + n) for n in range(8)]
    report = ContextBudgetEngine().synthesize(items, TokenBudget(total_budget=350), policy=policy, strict=True)
    assert report.over_budget is False and report.overflow_tokens == 0
    assert report.consumed_tokens <= report.effective_budget == 350
