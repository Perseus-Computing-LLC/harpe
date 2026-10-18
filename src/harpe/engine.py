"""Deterministic Context Budget Engine and Heuristic Token Counter."""

from __future__ import annotations

import math
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from .models import CognitiveDimension, ContextItem, PrunePolicy, SynthesisReport, TokenBudget
from .protocols import ContextSynthesizer, PruningHook, TokenCounter
from .routing import (
    QueryIntent,
    classify_query_intent,
    has_temporal_anchor,
    resolve_dynamic_budget,
)


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

    @staticmethod
    def _compute_item_utility(
        item: ContextItem,
        is_temporal: bool = False,
        temporal_boost: float = 1.15,
        now: Optional[float] = None,
    ) -> float:
        """Evaluate multi-attribute composite utility: 50% RRF/relevance, 30% Attention Density, 20% Exponential Recency."""
        curr_time = now if now is not None else time.time()
        delta_sec = max(0.0, curr_time - item.created_at)
        recency_decay = math.exp(-0.0001 * delta_sec)

        # Relevance / RRF component
        rrf = item.metadata.get("rrf_score")
        if rrf is None:
            rel = item.metadata.get("relevance")
            if rel is not None:
                try:
                    rrf = float(rel)
                except (ValueError, TypeError):
                    rrf = max(0.0, (100.0 - float(item.priority)) / 100.0)
            else:
                rrf = max(0.0, (100.0 - float(item.priority)) / 100.0)
        else:
            rrf = float(rrf)

        # Attention density component
        attn_density = float(item.metadata.get("attention_density", 0.5))

        utility = 100.0 * (0.50 * rrf + 0.30 * attn_density + 0.20 * recency_decay)
        if is_temporal and has_temporal_anchor(item.content, item.metadata):
            utility *= temporal_boost

        # Cognitive dimension weighting: active task goals and stances get momentum boost
        dim = item.dimension.value if isinstance(item.dimension, CognitiveDimension) else item.dimension
        if dim == CognitiveDimension.GOAL or dim == "goal":
            utility *= 1.25
        elif dim == CognitiveDimension.IDENTITY or dim == "identity":
            utility *= 1.10

        return max(0.01, utility)

    @classmethod
    def _solve_knapsack_dp(
        cls,
        items: List[ContextItem],
        utilities: List[float],
        capacity: int,
        dp_quant_bins: int = 1000,
        max_exact_items: int = 2000,
    ) -> Tuple[List[ContextItem], List[ContextItem]]:
        """Exact 0/1 Knapsack DP with quantized budget bins or Dantzig density fallback."""
        if not items or capacity <= 0:
            return [], list(items)

        n = len(items)
        if n > max_exact_items:
            # Dantzig greedy density sort (utility / tokens)
            paired = sorted(
                zip(items, utilities),
                key=lambda x: x[1] / max(1, x[0].token_count),
                reverse=True,
            )
            selected: List[ContextItem] = []
            pruned: List[ContextItem] = []
            acc = 0
            for it, _ in paired:
                if acc + it.token_count <= capacity:
                    selected.append(it)
                    acc += it.token_count
                else:
                    pruned.append(it)
            return selected, pruned

        # Quantize capacity to bounded integer bins
        scale = dp_quant_bins / capacity if capacity > dp_quant_bins else 1.0
        scaled_cap = int(math.ceil(capacity * scale))
        scaled_weights = [max(1, int(math.ceil(it.token_count * scale))) for it in items]

        dp = [0.0] * (scaled_cap + 1)
        parent = [[False] * (scaled_cap + 1) for _ in range(n)]

        for i in range(n):
            w = scaled_weights[i]
            v = utilities[i]
            for c in range(scaled_cap, w - 1, -1):
                if dp[c - w] + v > dp[c]:
                    dp[c] = dp[c - w] + v
                    parent[i][c] = True

        selected_indices = set()
        c = scaled_cap
        for i in range(n - 1, -1, -1):
            if parent[i][c]:
                selected_indices.add(i)
                c -= scaled_weights[i]

        selected_items = [items[i] for i in range(n) if i in selected_indices]
        pruned_items = [items[i] for i in range(n) if i not in selected_indices]
        return selected_items, pruned_items

    def synthesize(
        self,
        items: Sequence[Union[ContextItem, Dict[str, Any]]],
        budget: TokenBudget,
        policy: PrunePolicy = PrunePolicy.LOWEST_PRIORITY,
        destination: str = "remote",
        *,
        prune_policy: PrunePolicy | None = None,
        query: Optional[str] = None,
        intent: Optional[Union[QueryIntent, str]] = None,
        dynamic_budget: bool = False,
        aggregation_factor: float = 2.0,
        max_dynamic_budget: int = 4000,
        temporal_boost: float = 1.15,
        persona_item: Optional[Union[ContextItem, Dict[str, Any]]] = None,
    ) -> SynthesisReport:
        """``destination``: "remote" (default) withholds every item marked confidential, pinned or not, before any
        budgeting, so it cannot take a budget slot or appear in the output. "local" keeps everything.
        ``prune_policy``: alias for ``policy`` for ergonomics across benchmarks.
        ``query``: optional prompt/question string for automatic intent detection.
        ``intent``: explicit QueryIntent override (multi_session_aggregation, temporal, preference, factoid_single).
        ``dynamic_budget``: if True and intent is multi-session aggregation, expands knapsack budget capacity.
        ``temporal_boost``: utility weight multiplier (default 1.15x) for items with temporal anchors when query is temporal.
        ``persona_item``: optional compact user persona/preference slot injected at priority=0 when preference intent occurs.
        """
        if prune_policy is not None:
            policy = prune_policy
        if destination not in ("remote", "local"):
            raise ValueError("destination must be 'remote' or 'local'")

        # 1. Evaluate intent
        evaluated_intent: Optional[QueryIntent] = None
        if intent is not None:
            evaluated_intent = intent if isinstance(intent, QueryIntent) else QueryIntent(intent)
        elif query is not None:
            evaluated_intent = classify_query_intent(query)

        # 2. Dynamic budget expansion for multi-session aggregation
        dynamic_budget_expanded = False
        effective_budget = budget
        if dynamic_budget and evaluated_intent == QueryIntent.MULTI_SESSION_AGGREGATION:
            effective_budget = resolve_dynamic_budget(
                budget,
                evaluated_intent,
                aggregation_factor=aggregation_factor,
                max_budget=max_dynamic_budget,
            )
            dynamic_budget_expanded = True

        # 3. Item normalization and persona injection
        items_list: List[ContextItem] = []
        for it in items:
            if isinstance(it, dict):
                items_list.append(ContextItem.model_validate(it))
            else:
                items_list.append(it)

        if persona_item is not None:
            if isinstance(persona_item, dict):
                p_cand = ContextItem.model_validate(persona_item)
            else:
                p_cand = persona_item
            if not p_cand.pinned or p_cand.priority != 0:
                p_cand = p_cand.model_copy(update={"pinned": True, "priority": 0})
            if evaluated_intent == QueryIntent.PREFERENCE or evaluated_intent is None:
                items_list.insert(0, p_cand)

        withheld: List[ContextItem] = []
        if destination == "remote":
            withheld = [it for it in items_list if it.sensitivity == "confidential"]
            items_list = [it for it in items_list if it.sensitivity != "confidential"]
        resolved_items: List[ContextItem] = []
        for it in items_list:
            tcount = it.token_count
            if tcount <= 0:
                tcount = self.token_counter.count_tokens(it.content)
            resolved_items.append(it.model_copy(update={"token_count": tcount}))

        for hook in self.hooks:
            hook.pre_prune(resolved_items)

        available_capacity = effective_budget.available_context

        pinned_items: List[ContextItem] = []
        unpinned_items: List[ContextItem] = []

        for item in resolved_items:
            dim_val = item.dimension.value if isinstance(item.dimension, CognitiveDimension) else item.dimension
            is_invariant = dim_val == CognitiveDimension.INVARIANT or dim_val == "invariant"
            if item.pinned or item.priority == 0 or is_invariant:
                pinned_items.append(item)
            else:
                unpinned_items.append(item)

        def _calc_dim_breakdown(items_seq: List[ContextItem]) -> Tuple[Dict[str, int], Dict[str, int]]:
            counts: Dict[str, int] = {}
            toks: Dict[str, int] = {}
            for it in items_seq:
                d_key = str(it.dimension.value if isinstance(it.dimension, CognitiveDimension) else (it.dimension or "unspecified"))
                counts[d_key] = counts.get(d_key, 0) + 1
                toks[d_key] = toks.get(d_key, 0) + it.token_count
            return counts, toks

        pinned_tokens = sum(it.token_count for it in pinned_items)

        if pinned_tokens >= available_capacity:
            dim_counts, dim_tokens = _calc_dim_breakdown(pinned_items)
            report = SynthesisReport(
                included_items=pinned_items,
                pruned_items=unpinned_items,
                consumed_tokens=pinned_tokens,
                budget_remaining=0,
                pruned_tokens=sum(it.token_count for it in unpinned_items),
                withheld_items=withheld,
                policy_applied=policy,
                query_intent=evaluated_intent.value if evaluated_intent else None,
                dynamic_budget_expanded=dynamic_budget_expanded,
                dimension_counts=dim_counts,
                dimension_tokens=dim_tokens,
            )
            for hook in self.hooks:
                hook.post_prune(report)
            return report

        remaining_capacity = available_capacity - pinned_tokens

        included_unpinned: List[ContextItem] = []
        pruned_unpinned: List[ContextItem] = []

        is_temporal = evaluated_intent == QueryIntent.TEMPORAL

        if policy == PrunePolicy.LOWEST_PRIORITY:
            def priority_key(cand: ContextItem) -> tuple[float, float]:
                p = float(cand.priority)
                if is_temporal and has_temporal_anchor(cand.content, cand.metadata):
                    p = p / temporal_boost
                return (p, -cand.created_at)

            sorted_candidates = sorted(unpinned_items, key=priority_key)
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
                if is_temporal and has_temporal_anchor(item.content, item.metadata):
                    rel_float *= temporal_boost
                return (-rel_float, item.priority)

            sorted_candidates = sorted(unpinned_items, key=relevance_key)
            for cand in sorted_candidates:
                if cand.token_count <= remaining_capacity:
                    included_unpinned.append(cand)
                    remaining_capacity -= cand.token_count
                else:
                    pruned_unpinned.append(cand)
        elif policy == PrunePolicy.WARM_PREFIX:
            # Preserves early prompt-cached prefix items and recent active tail items,
            # evicting intermediate middle turns to maintain KV-cache reuse.
            chrono = sorted(unpinned_items, key=lambda x: x.created_at)
            prefix_budget = remaining_capacity // 2
            prefix_included: List[ContextItem] = []
            prefix_tokens = 0

            head_idx = 0
            while head_idx < len(chrono):
                cand = chrono[head_idx]
                if prefix_tokens + cand.token_count <= prefix_budget:
                    prefix_included.append(cand)
                    prefix_tokens += cand.token_count
                    head_idx += 1
                else:
                    break

            remaining_after_prefix = remaining_capacity - prefix_tokens
            tail_included: List[ContextItem] = []
            tail_tokens = 0
            tail_idx = len(chrono) - 1
            while tail_idx >= head_idx:
                cand = chrono[tail_idx]
                if tail_tokens + cand.token_count <= remaining_after_prefix:
                    tail_included.insert(0, cand)
                    tail_tokens += cand.token_count
                    tail_idx -= 1
                else:
                    break

            included_set = set(id(x) for x in prefix_included + tail_included)
            included_unpinned = prefix_included + tail_included
            pruned_unpinned = [x for x in chrono if id(x) not in included_set]
            remaining_capacity -= (prefix_tokens + tail_tokens)

        elif policy == PrunePolicy.KNAPSACK_DP:
            # 0/1 Knapsack Dynamic Programming with multi-attribute utility
            now = time.time()
            utilities = [
                self._compute_item_utility(it, is_temporal=is_temporal, temporal_boost=temporal_boost, now=now)
                for it in unpinned_items
            ]
            included_unpinned, pruned_unpinned = self._solve_knapsack_dp(
                unpinned_items, utilities, remaining_capacity
            )
            remaining_capacity -= sum(x.token_count for x in included_unpinned)

        elif policy == PrunePolicy.DUO_ATTENTION:
            # DuoAttention: Preserve attention sinks (head) + local sliding window (tail),
            # then apply Knapsack DP over intermediate retrieval candidates.
            chrono = sorted(unpinned_items, key=lambda x: x.created_at)
            k_sink_count = min(4, len(chrono))
            k_local_count = min(4, max(0, len(chrono) - k_sink_count))

            sinks: List[ContextItem] = chrono[:k_sink_count]
            local_window: List[ContextItem] = chrono[len(chrono) - k_local_count:] if k_local_count > 0 else []
            sink_and_local_ids = {it.id for it in sinks + local_window}
            middle_candidates = [it for it in chrono if it.id not in sink_and_local_ids]

            # Allocate budget for sinks and local window first
            guaranteed_items: List[ContextItem] = []
            for it in sinks + local_window:
                if it.token_count <= remaining_capacity:
                    guaranteed_items.append(it)
                    remaining_capacity -= it.token_count

            # Apply knapsack DP to select high-utility middle items
            now = time.time()
            utilities = [
                self._compute_item_utility(it, is_temporal=is_temporal, temporal_boost=temporal_boost, now=now)
                for it in middle_candidates
            ]
            knapsack_selected, knapsack_pruned = self._solve_knapsack_dp(
                middle_candidates, utilities, remaining_capacity
            )
            remaining_capacity -= sum(x.token_count for x in knapsack_selected)

            included_unpinned = guaranteed_items + knapsack_selected
            included_ids = {it.id for it in included_unpinned}
            pruned_unpinned = [it for it in unpinned_items if it.id not in included_ids]

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
        dim_counts, dim_tokens = _calc_dim_breakdown(final_included)

        report = SynthesisReport(
            included_items=final_included,
            pruned_items=final_pruned,
            consumed_tokens=total_consumed,
            budget_remaining=max(0, available_capacity - total_consumed),
            pruned_tokens=total_pruned,
            withheld_items=withheld,
            policy_applied=policy,
            query_intent=evaluated_intent.value if evaluated_intent else None,
            dynamic_budget_expanded=dynamic_budget_expanded,
            dimension_counts=dim_counts,
            dimension_tokens=dim_tokens,
        )

        for hook in self.hooks:
            hook.post_prune(report)

        return report
