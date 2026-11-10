"""severity: medium
finding: with hooks, synthesize gives pre_prune shallow copies, so a hook's in-place edit of an item's metadata is written into the caller's item and replayed into later calls

ContextBudgetEngine.synthesize (src/harpe/engine.py:511) copies the items before the ``pre_prune`` hooks so that
hooks work on copies: ``_count_uncounted`` (engine.py:366-376) says "Hooks may change items in place, so with hooks
every item is copied", and the comment after the hooks (engine.py:571-572) says "Hooks work on copies and may
reclassify or rewrite them". The copy is ``it.model_copy()`` (engine.py:374; also ``model_copy(update=...)`` at
engine.py:372), which pydantic makes shallow: the copy's ``metadata`` is the caller's dict object. A hook that
writes a top-level field (``it.priority = ...``, ``it.sensitivity = ...``) changes only its copy, but a hook that
writes into ``it.metadata`` - the place harpe itself reads ``relevance``, ``rrf_score`` and
``attention_density`` from (engine.py:151-165, 421-430) - writes into the caller's item. The edit outlives the call: the
next synthesize over the same items, with another engine or another hook set, selects on what the earlier call's
hook wrote, not on the caller's data. The copy exists to stop exactly this write; it stops it for scalar fields
only. This is the same time-of-check/time-of-use seam as round 1's F2 (hooks on copies), on the nested fields.

The tests use a per-query relevance hook (it scores the items for one query) and accept any fix that leaves the
caller's items as they were given (for example a deep copy, or a copy of ``metadata``).
"""
from __future__ import annotations

from typing import List, Sequence

from harpe import ContextBudgetEngine, ContextItem, PrunePolicy, TokenBudget


class ScoreForQuery:
    """A pre_prune hook that scores relevance for one query: the item named ``favoured`` is the relevant one."""

    def __init__(self, favoured: str) -> None:
        self.favoured = favoured

    def pre_prune(self, items: Sequence[ContextItem]) -> None:
        for it in items:
            it.metadata["relevance"] = 1.0 if it.id == self.favoured else 0.0

    def post_prune(self, report) -> None:
        pass


class DemoteAll:
    """A pre_prune hook that only writes a top-level field of each item."""

    def pre_prune(self, items: Sequence[ContextItem]) -> None:
        for it in items:
            it.priority = 99

    def post_prune(self, report) -> None:
        pass


def _items() -> List[ContextItem]:
    return [
        ContextItem(id="runbook", content="Restart the ingest worker, then drain the queue.", token_count=20,
                    metadata={"relevance": 0.9}),
        ContextItem(id="trivia", content="The office plant is a ficus.", token_count=20,
                    metadata={"relevance": 0.1}),
    ]


def _budget() -> TokenBudget:
    return TokenBudget(total_budget=25)  # room for one of the two items


def test_control_a_hook_that_writes_a_top_level_field_leaves_the_callers_item_alone():
    items = _items()
    report = ContextBudgetEngine(hooks=[DemoteAll()]).synthesize(items, _budget(), policy=PrunePolicy.LEAST_RELEVANT)
    assert {it.priority for it in report.included_items + report.pruned_items} == {99}
    assert [it.priority for it in items] == [50, 50]


def test_control_without_hooks_the_callers_relevance_picks_the_runbook():
    report = ContextBudgetEngine().synthesize(_items(), _budget(), policy=PrunePolicy.LEAST_RELEVANT)
    assert [it.id for it in report.included_items] == ["runbook"]


def test_a_hook_editing_metadata_does_not_change_the_callers_item():
    items = _items()
    ContextBudgetEngine(hooks=[ScoreForQuery("trivia")]).synthesize(items, _budget(), policy=PrunePolicy.LEAST_RELEVANT)
    assert [it.metadata for it in items] == [{"relevance": 0.9}, {"relevance": 0.1}], (
        f"the hook's per-call scores were written into the caller's items: {[it.metadata for it in items]}"
    )


def test_a_later_call_without_the_hook_selects_on_the_callers_data_not_on_the_earlier_hooks_scores():
    items = _items()
    ContextBudgetEngine(hooks=[ScoreForQuery("trivia")]).synthesize(items, _budget(), policy=PrunePolicy.LEAST_RELEVANT)
    later = ContextBudgetEngine().synthesize(items, _budget(), policy=PrunePolicy.LEAST_RELEVANT)
    assert [it.id for it in later.included_items] == ["runbook"], (
        f"a hook-less call selected {[it.id for it in later.included_items]} on an earlier hook's scores"
    )
