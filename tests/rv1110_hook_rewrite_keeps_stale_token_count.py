"""severity: medium
finding: a pre_prune hook that rewrites an item's content keeps the pre-hook token count, so synthesize includes far more than the budget and reports it as within budget

Round 1 (F2) moved the withholding decision after the hooks and added a re-count, and the engine now states
(src/harpe/engine.py:582-584): "Hooks work on copies and may reclassify, rewrite or add to them: what is withheld,
that each id is unique and what each item costs are decided again on the items they left, so the items selected are
the items checked." The re-count (`_count_uncounted`, engine.py:370-383, called at :589) only counts items whose
`token_count <= 0`. Every item was already counted at :576, BEFORE the hooks ran, so an item a hook rewrote keeps the
count of the text it no longer holds. Sensitivity is re-checked after the hooks; cost is not: same time-of-check /
time-of-use class as F2, other field (R1 listed it under "Suspected").

SECURITY.md (Scope): "Unpinned items are added only while they fit the budget's available context."
"""
from harpe import ContextBudgetEngine, ContextItem, TokenBudget


class _Expand:
    """A hook that enriches each item (e.g. resolves a reference into the text it stands for)."""

    def pre_prune(self, items):
        for it in items:
            it.content = it.content + " resolved detail" * 400

    def post_prune(self, report):
        pass


class _Nothing:
    def pre_prune(self, items):
        pass

    def post_prune(self, report):
        pass


def _items():
    return [ContextItem(id=f"i{n}", content=f"see document {n}", priority=50) for n in range(5)]


def test_control_without_a_rewrite_the_report_counts_what_it_includes():
    engine = ContextBudgetEngine(hooks=[_Nothing()])
    budget = TokenBudget(total_budget=1000)
    report = engine.synthesize(_items(), budget, destination="local")
    actual = sum(engine.token_counter.count_tokens(it.content) for it in report.included_items)
    assert len(report.included_items) == 5
    assert report.consumed_tokens == actual <= budget.available_context


def test_items_a_hook_rewrote_are_costed_as_rewritten():
    engine = ContextBudgetEngine(hooks=[_Expand()])
    budget = TokenBudget(total_budget=1000)
    report = engine.synthesize(_items(), budget, destination="local")
    actual = sum(engine.token_counter.count_tokens(it.content) for it in report.included_items)
    # Each rewritten item is ~800+ tokens; five of them cannot fit a 1000-token budget.
    assert report.consumed_tokens == actual, (
        f"report says {report.consumed_tokens} tokens consumed, the included content is {actual} tokens"
    )
    assert actual <= budget.available_context, (
        f"included content is {actual} tokens against an available context of {budget.available_context}"
    )
