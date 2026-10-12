"""Deterministic Context Budget Engine and Heuristic Token Counter."""

from __future__ import annotations

import re
from typing import List, Sequence
from .models import ContextItem, PrunePolicy, SynthesisReport, TokenBudget
from .protocols import ContextSynthesizer, PruningHook, TokenCounter


class HeuristicTokenCounter:
    """Deterministic token counter based on whitespace, word boundaries, and character length."""

    _WORD_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        matches = self._WORD_RE.findall(text)
        token_count = len(matches)
        char_fallback = max(1, (len(text) + 3) // 4)
        return max(token_count, char_fallback)


class ContextBudgetEngine(ContextSynthesizer):
    """Deterministic context synthesis engine enforcing explicit token budgets."""

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        hooks: Sequence[PruningHook] | None = None,
    ) -> None:
        self.token_counter = token_counter or HeuristicTokenCounter()
        self.hooks: List[PruningHook] = list(hooks or [])

    def register_hook(self, hook: PruningHook) -> None:
        self.hooks.append(hook)

    def synthesize(
        self,
        items: Sequence[ContextItem],
        budget: TokenBudget,
        policy: PrunePolicy = PrunePolicy.LOWEST_PRIORITY,
        destination: str = "remote",
    ) -> SynthesisReport:
        """``destination``: "remote" (default) withholds every item marked confidential, pinned or not, before any
        budgeting, so it cannot take a budget slot or appear in the output. "local" keeps everything."""
        if destination not in ("remote", "local"):
            raise ValueError("destination must be 'remote' or 'local'")
        withheld: List[ContextItem] = []
        if destination == "remote":
            withheld = [it for it in items if it.sensitivity == "confidential"]
            items = [it for it in items if it.sensitivity != "confidential"]
        resolved_items: List[ContextItem] = []
        for it in items:
            tcount = it.token_count
            if tcount <= 0:
                tcount = self.token_counter.count_tokens(it.content)
            resolved_items.append(it.model_copy(update={"token_count": tcount}))

        for hook in self.hooks:
            hook.pre_prune(resolved_items)

        available_capacity = budget.available_context

        pinned_items: List[ContextItem] = []
        unpinned_items: List[ContextItem] = []

        for item in resolved_items:
            if item.pinned or item.priority == 0:
                pinned_items.append(item)
            else:
                unpinned_items.append(item)

        pinned_tokens = sum(it.token_count for it in pinned_items)

        if pinned_tokens >= available_capacity:
            report = SynthesisReport(
                included_items=pinned_items,
                pruned_items=unpinned_items,
                consumed_tokens=pinned_tokens,
                budget_remaining=0,
                pruned_tokens=sum(it.token_count for it in unpinned_items),
                withheld_items=withheld,
                policy_applied=policy,
            )
            for hook in self.hooks:
                hook.post_prune(report)
            return report

        remaining_capacity = available_capacity - pinned_tokens

        included_unpinned: List[ContextItem] = []
        pruned_unpinned: List[ContextItem] = []

        if policy == PrunePolicy.LOWEST_PRIORITY:
            sorted_candidates = sorted(
                unpinned_items,
                key=lambda x: (x.priority, -x.created_at),
            )
            for cand in sorted_candidates:
                if cand.token_count <= remaining_capacity:
                    included_unpinned.append(cand)
                    remaining_capacity -= cand.token_count
                else:
                    pruned_unpinned.append(cand)

        elif policy == PrunePolicy.FIFO:
            sorted_candidates = sorted(
                unpinned_items,
                key=lambda x: -x.created_at,
            )
            for cand in sorted_candidates:
                if cand.token_count <= remaining_capacity:
                    included_unpinned.append(cand)
                    remaining_capacity -= cand.token_count
                else:
                    pruned_unpinned.append(cand)

        elif policy == PrunePolicy.TRUNCATE_TAIL:
            for cand in unpinned_items:
                if cand.token_count <= remaining_capacity:
                    included_unpinned.append(cand)
                    remaining_capacity -= cand.token_count
                else:
                    pruned_unpinned.append(cand)

        elif policy == PrunePolicy.LEAST_RELEVANT:
            def relevance_key(item: ContextItem) -> tuple[float, int]:
                rel = item.metadata.get("relevance", 0.0)
                try:
                    rel_float = float(rel)
                except (ValueError, TypeError):
                    rel_float = 0.0
                return (-rel_float, item.priority)

            sorted_candidates = sorted(unpinned_items, key=relevance_key)
            for cand in sorted_candidates:
                if cand.token_count <= remaining_capacity:
                    included_unpinned.append(cand)
                    remaining_capacity -= cand.token_count
                else:
                    pruned_unpinned.append(cand)
        else:
            sorted_candidates = sorted(unpinned_items, key=lambda x: x.priority)
            for cand in sorted_candidates:
                if cand.token_count <= remaining_capacity:
                    included_unpinned.append(cand)
                    remaining_capacity -= cand.token_count
                else:
                    pruned_unpinned.append(cand)

        included_set = {it.id for it in included_unpinned}
        final_unpinned_in_order = [it for it in unpinned_items if it.id in included_set]

        final_included_set = {it.id for it in pinned_items}.union(included_set)
        final_included = [it for it in resolved_items if it.id in final_included_set]

        final_pruned_set = {it.id for it in resolved_items if it.id not in final_included_set}
        final_pruned = [it for it in resolved_items if it.id in final_pruned_set]

        total_consumed = sum(it.token_count for it in final_included)
        total_pruned = sum(it.token_count for it in final_pruned)

        report = SynthesisReport(
            included_items=final_included,
            pruned_items=final_pruned,
            consumed_tokens=total_consumed,
            budget_remaining=max(0, available_capacity - total_consumed),
            pruned_tokens=total_pruned,
            withheld_items=withheld,
            policy_applied=policy,
        )

        for hook in self.hooks:
            hook.post_prune(report)

        return report
