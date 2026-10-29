"""Deterministic Context Budget Engine and Heuristic Token Counter."""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union
from .models import CognitiveDimension, ContextItem, PrunePolicy, SynthesisReport, TokenBudget
from .protocols import ContextSynthesizer, PruningHook, TokenCounter
from .routing import (
    QueryIntent,
    RouteDecision,
    classify_query_intent,
    has_temporal_anchor,
    resolve_dynamic_budget,
    route_by_fit,
)


class HeuristicTokenCounter:
    """Default token estimate, the same in Rust and Python: the larger of (pieces + hex extras) and a quarter of
    the characters, rounded up. A piece is a run of ASCII letters, digits and underscores, or any single other
    non-whitespace character: an ASCII symbol or any non-ASCII code point, so CJK text, emoji and accented letters
    count one per code point instead of joining a word. Every maximal run of 8 or more hex digits adds
    ``len // 3`` extras on top of the piece it sits in, so hashes and ids cost about one token per three
    characters. Whitespace is what ``str.isspace()`` accepts. Both test suites check ``tests/token_vectors.tsv``.

    Measured against tiktoken ``o200k_base`` on the shared vectors this undercounts emoji and hex ids (about 0.6x)
    and overcounts CJK (about 1.3-1.4x) and English prose (about 1.2x); 1.11x over all 208 vectors. It is an
    estimate, not a tokenizer. Use :class:`harpe.tokenizers.TiktokenCounter` or your model's tokenizer for exact
    counts."""

    _PIECE_RE = re.compile(r"[A-Za-z0-9_]+|\S")
    _HEX_RUN_RE = re.compile(r"[0-9a-fA-F]{8,}")

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        pieces = len(self._PIECE_RE.findall(text))
        pieces += sum(len(run) // 3 for run in self._HEX_RUN_RE.findall(text))
        return max(pieces, (len(text) + 3) // 4)


class BudgetOverflowError(ValueError):
    """Raised by :meth:`ContextBudgetEngine.synthesize` with ``strict=True`` when the pinned, priority-0 and
    invariant items alone exceed the available context. Carries the report that would otherwise have been
    returned as ``report``."""

    def __init__(self, report: SynthesisReport) -> None:
        self.report = report
        super().__init__(
            f"pinned items use {report.consumed_tokens} tokens against a ceiling of {report.effective_budget}: "
            f"{report.overflow_tokens} over"
        )


class ContextBudgetEngine(ContextSynthesizer):
    """Deterministic context synthesis engine enforcing explicit token budgets."""

    def __init__(
        self,
        token_counter: TokenCounter | None = None,
        hooks: Sequence[PruningHook] | None = None,
    ) -> None:
        self.token_counter = token_counter or HeuristicTokenCounter()
        self.hooks: List[PruningHook] = list(hooks or [])

    def route(self, items: Iterable[Union[ContextItem, str]], threshold_tokens: Optional[int]) -> RouteDecision:
        """:func:`harpe.routing.route_by_fit` with this engine's token counter: whether to hand a whole
        history over (it fits under ``threshold_tokens``) or to retrieve and :meth:`synthesize`."""
        return route_by_fit(items, threshold_tokens, self.token_counter)

    def register_hook(self, hook: PruningHook) -> None:
        self.hooks.append(hook)

    @staticmethod
    def _compute_item_utility(
        item: ContextItem,
        is_temporal: bool = False,
        temporal_boost: float = 1.15,
        now: Optional[float] = None,
    ) -> float:
        """Evaluate multi-attribute composite utility: 50% RRF/relevance, 30% Attention Density, 20% Exponential Recency.

        ``now`` is the reference time the recency decay is measured from. It is never the wall clock: the engine
        passes ``max(created_at)`` over the inputs so the same items always score the same, and with no ``now``
        the item is treated as current (decay 1.0)."""
        curr_time = now if now is not None else item.created_at
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

    @staticmethod
    def _exact_knapsack(weights: List[int], utilities: List[float], capacity: int, min_block: int = 16) -> set:
        """Exact 0/1 knapsack over ``weights``/``utilities`` (every weight in ``1..capacity``); returns the set of
        chosen indices.

        This computes exactly the textbook one-row DP, ``dp[c] = max(dp[c], dp[c - w] + u)`` for every item in
        input order and every cell ``c``, with an item recorded as taken at ``c`` only when it strictly improves
        the cell, so ties resolve as they always did (earlier items win). What changes is how many cells are
        visited one at a time:

        * Cells never consulted are skipped: for item ``i`` only ``c <= min(capacity, sum(w[:i+1]))`` can be
          reached, and only ``c >= capacity - sum(w[i+1:])`` can lie on the final backtrace path. Above the
          reachable prefix the row is flat, so those cells are filled by one slice assignment.
        * ``dp`` is non-decreasing in ``c``, so for a block of cells ``[a, b]`` two comparisons decide whole
          blocks: ``dp[b - w] + u <= dp[a]`` means no cell in the block improves, and ``dp[a - w] + u > dp[b]``
          means every cell does, in which case the new values come from one list slice. Both tests use the
          same float addition as the cell update, so they never disagree with it. Blocks that are neither are
          halved until they are shorter than ``min_block`` and then scanned cell by cell.

        Memory is one ``bytearray`` of ``capacity + 1`` keep flags per item, as before.
        """
        n = len(weights)
        dp = [0.0] * (capacity + 1)
        keep = [bytearray(capacity + 1) for _ in range(n)]
        suffix = [0] * (n + 1)
        for i in range(n - 1, -1, -1):
            suffix[i] = suffix[i + 1] + weights[i]
        prefix = 0
        for i in range(n):
            w, u, row = weights[i], utilities[i], keep[i]
            prefix += w
            hi = min(capacity, prefix)
            lo = max(w, capacity - suffix[i + 1])
            if lo <= hi:
                # Top-down so that reads of dp[c - w] always see the row before this item (0/1, not unbounded).
                stack = [(lo, hi)]
                while stack:
                    a, b = stack.pop()
                    if dp[b - w] + u <= dp[a]:
                        continue
                    if dp[a - w] + u > dp[b]:
                        dp[a:b + 1] = [z + u for z in dp[a - w:b - w + 1]]
                        row[a:b + 1] = b"\x01" * (b - a + 1)
                        continue
                    if b - a < min_block:
                        for c in range(b, a - 1, -1):
                            cand = dp[c - w] + u
                            if cand > dp[c]:
                                dp[c] = cand
                                row[c] = 1
                        continue
                    mid = (a + b) >> 1
                    stack.append((a, mid))
                    stack.append((mid + 1, b))
            if hi < capacity:
                # Every cell above the reachable prefix still holds the previous row's dp[prefix] (the block
                # phase never writes above hi); item i improves all of those cells or none of them.
                top_old = dp[capacity]
                top_cand = top_old + u
                span = capacity - hi
                if top_cand > top_old:
                    dp[hi + 1:] = [top_cand] * span
                    row[hi + 1:] = b"\x01" * span
                else:
                    dp[hi + 1:] = [top_old] * span
        take = set()
        c = capacity
        for i in range(n - 1, -1, -1):
            if keep[i][c]:
                take.add(i)
                c -= weights[i]
        return take

    @classmethod
    def _solve_knapsack_dp(
        cls,
        items: List[ContextItem],
        utilities: List[float],
        capacity: int,
        max_cells: int = 5_000_000,
    ) -> Tuple[List[ContextItem], List[ContextItem]]:
        """Select items of maximum total utility whose token counts sum to at most ``capacity``.

        Items heavier than the whole capacity are rejected up front: they could never be chosen and their DP
        rows would change nothing. When the remaining weights share a common divisor the weights and the
        capacity are divided by it, which is lossless (every reachable total is a multiple of it) and shrinks
        the table. Then, when ``n * (capacity + 1) <= max_cells`` (default 5,000,000) on those reduced sizes,
        :meth:`_exact_knapsack` runs the exact 0/1 DP on the raw token weights, using n*(capacity+1) bytes of
        keep flags. Above that size it falls back to utility-density greedy followed by a slack-fill pass in
        utility order. Neither path quantises the budget, so with positive utilities the unused capacity is
        smaller than the largest single token count, not n times a bin size. Returns the chosen and the
        rejected items, each in input order.
        """
        n = len(items)
        if n == 0 or capacity <= 0:
            return [], list(items)
        weights = [max(1, it.token_count) for it in items]

        fit = [i for i in range(n) if weights[i] <= capacity]
        if not fit:
            return [], list(items)
        divisor = math.gcd(*(weights[i] for i in fit))
        reduced_capacity = capacity // divisor

        if len(fit) * (reduced_capacity + 1) <= max_cells:
            chosen = cls._exact_knapsack(
                [weights[i] // divisor for i in fit], [utilities[i] for i in fit], reduced_capacity
            )
            take = {fit[k] for k in chosen}
        else:
            # Dantzig density order, then fill whatever slack is left in plain utility order: density alone
            # can skip a small high-utility item that would still have fit.
            order = sorted(range(n), key=lambda i: (-utilities[i] / weights[i], i))
            take, used = set(), 0
            for i in order:
                if used + weights[i] <= capacity:
                    take.add(i)
                    used += weights[i]
            for i in sorted(range(n), key=lambda i: (-utilities[i], i)):
                if i not in take and used + weights[i] <= capacity:
                    take.add(i)
                    used += weights[i]

        selected_items = [items[i] for i in range(n) if i in take]
        pruned_items = [items[i] for i in range(n) if i not in take]
        return selected_items, pruned_items

    @staticmethod
    def _dim_breakdown(items_seq: Sequence[ContextItem]) -> Tuple[Dict[str, int], Dict[str, int]]:
        counts: Dict[str, int] = {}
        toks: Dict[str, int] = {}
        for it in items_seq:
            d_key = str(it.dimension.value if isinstance(it.dimension, CognitiveDimension) else (it.dimension or "unspecified"))
            counts[d_key] = counts.get(d_key, 0) + 1
            toks[d_key] = toks.get(d_key, 0) + it.token_count
        return counts, toks

    @staticmethod
    def _is_never_pruned(item: ContextItem) -> bool:
        """Pinned, priority-0 and invariant-dimension items bypass selection and are always included."""
        dim_val = item.dimension.value if isinstance(item.dimension, CognitiveDimension) else item.dimension
        return item.pinned or item.priority == 0 or dim_val in (CognitiveDimension.INVARIANT, "invariant")

    def _normalise_items(
        self,
        items: Sequence[Union[ContextItem, Dict[str, Any]]],
        persona_item: Optional[Union[ContextItem, Dict[str, Any]]],
        evaluated_intent: Optional[QueryIntent],
    ) -> List[ContextItem]:
        """Validate dicts, inject the persona slot at priority 0 for preference (or unknown) intent, refuse
        duplicate ids."""
        items_list = [ContextItem.model_validate(it) if isinstance(it, dict) else it for it in items]
        if persona_item is not None:
            p_cand = ContextItem.model_validate(persona_item) if isinstance(persona_item, dict) else persona_item
            if not p_cand.pinned or p_cand.priority != 0:
                p_cand = p_cand.model_copy(update={"pinned": True, "priority": 0})
            if evaluated_intent == QueryIntent.PREFERENCE or evaluated_intent is None:
                items_list.insert(0, p_cand)
        # Ids come straight from the caller and are only unique by convention. Selection is tracked by object
        # identity, so duplicates cannot smuggle a pruned item back in, but a report keyed by id would still be
        # ambiguous for the caller. Refuse them rather than guess.
        ids = [it.id for it in items_list]
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise ValueError(f"duplicate ContextItem ids: {dupes}")
        return items_list

    def _count_uncounted(self, items_list: List[ContextItem]) -> List[ContextItem]:
        """Only items without a count are counted and copied. Hooks may change items in place, so with hooks
        every item is copied."""
        resolved: List[ContextItem] = []
        for it in items_list:
            if it.token_count <= 0:
                it = it.model_copy(update={"token_count": self.token_counter.count_tokens(it.content)})
            elif self.hooks:
                it = it.model_copy()
            resolved.append(it)
        return resolved

    @staticmethod
    def _take_in_order(ordered: Sequence[ContextItem], capacity: int) -> Tuple[List[ContextItem], List[ContextItem]]:
        """Greedy pass: include each item in the given order while it fits, prune the rest."""
        included: List[ContextItem] = []
        pruned: List[ContextItem] = []
        for cand in ordered:
            if cand.token_count <= capacity:
                included.append(cand)
                capacity -= cand.token_count
            else:
                pruned.append(cand)
        return included, pruned

    def _select_unpinned(
        self,
        policy: PrunePolicy,
        items: List[ContextItem],
        capacity: int,
        *,
        is_temporal: bool = False,
        temporal_boost: float = 1.15,
        reference_time: Optional[float] = None,
    ) -> Tuple[List[ContextItem], List[ContextItem]]:
        """Choose which prunable items fit in ``capacity`` under ``policy``. Returns ``(included, pruned)``;
        every input appears in exactly one of the two, and ``sum(token_count for included) <= capacity``.
        Raises ``ValueError`` for a policy this engine does not implement rather than silently falling back
        to LOWEST_PRIORITY, which used to hide a misconfigured caller behind a plausible result."""
        if policy == PrunePolicy.LOWEST_PRIORITY:
            def priority_key(cand: ContextItem) -> tuple[float, float]:
                p = float(cand.priority)
                if is_temporal and has_temporal_anchor(cand.content, cand.metadata):
                    p = p / temporal_boost
                return (p, -cand.created_at)

            return self._take_in_order(sorted(items, key=priority_key), capacity)

        if policy == PrunePolicy.FIFO:
            return self._take_in_order(sorted(items, key=lambda x: -x.created_at), capacity)

        if policy == PrunePolicy.TRUNCATE_TAIL:
            return self._take_in_order(items, capacity)

        if policy == PrunePolicy.LEAST_RELEVANT:
            def relevance_key(item: ContextItem) -> tuple[float, int]:
                try:
                    rel_float = float(item.metadata.get("relevance", 0.0))
                except (ValueError, TypeError):
                    rel_float = 0.0
                if is_temporal and has_temporal_anchor(item.content, item.metadata):
                    rel_float *= temporal_boost
                return (-rel_float, item.priority)

            return self._take_in_order(sorted(items, key=relevance_key), capacity)

        if policy == PrunePolicy.WARM_PREFIX:
            # Keep the early prompt-cached prefix (up to half the capacity) and the most recent tail, evicting
            # the middle turns, so the KV cache for the prefix stays reusable.
            chrono = sorted(items, key=lambda x: x.created_at)
            prefix_budget = capacity // 2
            prefix: List[ContextItem] = []
            prefix_tokens = 0
            head_idx = 0
            while head_idx < len(chrono) and prefix_tokens + chrono[head_idx].token_count <= prefix_budget:
                prefix.append(chrono[head_idx])
                prefix_tokens += chrono[head_idx].token_count
                head_idx += 1
            tail: List[ContextItem] = []
            tail_tokens = 0
            tail_idx = len(chrono) - 1
            while tail_idx >= head_idx and tail_tokens + chrono[tail_idx].token_count <= capacity - prefix_tokens:
                tail.insert(0, chrono[tail_idx])
                tail_tokens += chrono[tail_idx].token_count
                tail_idx -= 1
            included = prefix + tail
            kept = {id(x) for x in included}
            return included, [x for x in chrono if id(x) not in kept]

        if policy == PrunePolicy.KNAPSACK_DP:
            utilities = [
                self._compute_item_utility(it, is_temporal=is_temporal, temporal_boost=temporal_boost, now=reference_time)
                for it in items
            ]
            return self._solve_knapsack_dp(items, utilities, capacity)

        if policy == PrunePolicy.DUO_ATTENTION:
            # DuoAttention: attention sinks (oldest four) and the local window (newest four) are taken first in
            # order while they fit; the middle is then solved as a knapsack.
            chrono = sorted(items, key=lambda x: x.created_at)
            k_sink = min(4, len(chrono))
            k_local = min(4, max(0, len(chrono) - k_sink))
            guaranteed_candidates = chrono[:k_sink] + (chrono[len(chrono) - k_local:] if k_local > 0 else [])
            sink_and_local = {id(it) for it in guaranteed_candidates}
            middle = [it for it in chrono if id(it) not in sink_and_local]

            guaranteed: List[ContextItem] = []
            for it in guaranteed_candidates:
                if it.token_count <= capacity:
                    guaranteed.append(it)
                    capacity -= it.token_count

            utilities = [
                self._compute_item_utility(it, is_temporal=is_temporal, temporal_boost=temporal_boost, now=reference_time)
                for it in middle
            ]
            chosen_middle, _ = self._solve_knapsack_dp(middle, utilities, capacity)
            included = guaranteed + chosen_middle
            kept = {id(it) for it in included}
            return included, [it for it in items if id(it) not in kept]

        raise ValueError(f"unknown prune policy: {policy!r}")

    @staticmethod
    def _resolve_intent_and_budget(
        budget: TokenBudget,
        query: Optional[str],
        intent: Optional[Union[QueryIntent, str]],
        dynamic_budget: bool,
        aggregation_factor: float,
        max_dynamic_factor: float,
    ) -> Tuple[Optional[QueryIntent], TokenBudget, bool]:
        """Explicit intent wins over query classification; the budget is widened only for aggregation."""
        evaluated: Optional[QueryIntent] = None
        if intent is not None:
            evaluated = intent if isinstance(intent, QueryIntent) else QueryIntent(intent)
        elif query is not None:
            evaluated = classify_query_intent(query)
        expanded = dynamic_budget and evaluated == QueryIntent.MULTI_SESSION_AGGREGATION
        if expanded:
            budget = resolve_dynamic_budget(
                budget, evaluated, aggregation_factor=aggregation_factor, max_factor=max_dynamic_factor
            )
        return evaluated, budget, expanded

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
        max_dynamic_factor: float = 2.0,
        temporal_boost: float = 1.15,
        persona_item: Optional[Union[ContextItem, Dict[str, Any]]] = None,
        strict: bool = False,
        now: Optional[float] = None,
    ) -> SynthesisReport:
        """``destination``: "remote" (default) withholds every item marked confidential, pinned or not, before any
        budgeting, so it cannot take a budget slot or appear in the output. "local" keeps everything.
        ``prune_policy``: alias for ``policy`` for ergonomics across benchmarks. An unknown policy raises
        ``ValueError``.
        ``query``: optional prompt/question string for automatic intent detection.
        ``intent``: explicit QueryIntent override (multi_session_aggregation, temporal, preference, factoid_single).
        ``dynamic_budget``: if True and intent is multi-session aggregation, widens ``total_budget`` by
        ``min(aggregation_factor, max_dynamic_factor)`` (never below 1.0, so never smaller than ``budget``). The
        ceiling actually used is reported as ``SynthesisReport.effective_budget`` on every path.
        ``temporal_boost``: utility weight multiplier (default 1.15x) for items with temporal anchors when query is temporal.
        ``persona_item``: optional compact user persona/preference slot injected at priority=0 when preference intent occurs.
        ``strict``: pinned, priority-0 and invariant-dimension items are never pruned, so when they alone exceed the
        ceiling the report comes back with ``over_budget=True`` and ``overflow_tokens`` set. With ``strict=True``
        that case raises :class:`BudgetOverflowError` (carrying the report) instead of returning.
        ``now``: reference time for the recency term of ``KNAPSACK_DP`` and ``DUO_ATTENTION``. Defaults to the
        newest ``created_at`` among the items, never the wall clock, so the same inputs give the same selection
        whenever they are replayed. The value used is reported as ``SynthesisReport.reference_time``.
        """
        if prune_policy is not None:
            policy = prune_policy
        if destination not in ("remote", "local"):
            raise ValueError("destination must be 'remote' or 'local'")
        if policy not in PrunePolicy.__members__.values():
            raise ValueError(f"unknown prune policy: {policy!r}")

        evaluated_intent, effective_budget, dynamic_budget_expanded = self._resolve_intent_and_budget(
            budget, query, intent, dynamic_budget, aggregation_factor, max_dynamic_factor
        )
        available_capacity = effective_budget.available_context

        items_list = self._normalise_items(items, persona_item, evaluated_intent)
        withheld: List[ContextItem] = []
        if destination == "remote":
            withheld = [it for it in items_list if it.sensitivity == "confidential"]
            items_list = [it for it in items_list if it.sensitivity != "confidential"]
        resolved_items = self._count_uncounted(items_list)
        for hook in self.hooks:
            hook.pre_prune(resolved_items)

        # Recency is measured from the newest item, not from the wall clock, so a replay scores identically.
        reference_time = now
        if reference_time is None and resolved_items:
            reference_time = max(it.created_at for it in resolved_items)

        pinned_items = [it for it in resolved_items if self._is_never_pruned(it)]
        unpinned_items = [it for it in resolved_items if not self._is_never_pruned(it)]
        pinned_tokens = sum(it.token_count for it in pinned_items)

        if pinned_tokens >= available_capacity:
            # Nothing else fits. Pinned items are kept even over the ceiling, so say so in the report rather
            # than hiding it behind budget_remaining=0; strict callers get an exception instead.
            overflow = pinned_tokens - available_capacity
            final_included, final_pruned = pinned_items, unpinned_items
        else:
            overflow = 0
            included_unpinned, _ = self._select_unpinned(
                policy,
                unpinned_items,
                available_capacity - pinned_tokens,
                is_temporal=evaluated_intent == QueryIntent.TEMPORAL,
                temporal_boost=temporal_boost,
                reference_time=reference_time,
            )
            # Rebuild input order from the objects each policy chose, never from ids: id membership would
            # re-admit every item sharing an id with a selected one and break the hard ceiling.
            chosen = {id(it) for it in pinned_items} | {id(it) for it in included_unpinned}
            final_included = [it for it in resolved_items if id(it) in chosen]
            final_pruned = [it for it in resolved_items if id(it) not in chosen]

        total_consumed = sum(it.token_count for it in final_included)
        dim_counts, dim_tokens = self._dim_breakdown(final_included)
        report = SynthesisReport(
            included_items=final_included,
            pruned_items=final_pruned,
            consumed_tokens=total_consumed,
            budget_remaining=max(0, available_capacity - total_consumed),
            pruned_tokens=sum(it.token_count for it in final_pruned),
            withheld_items=withheld,
            policy_applied=policy,
            query_intent=evaluated_intent.value if evaluated_intent else None,
            dynamic_budget_expanded=dynamic_budget_expanded,
            over_budget=overflow > 0,
            overflow_tokens=overflow,
            effective_budget=available_capacity,
            reference_time=reference_time,
            dimension_counts=dim_counts,
            dimension_tokens=dim_tokens,
        )
        if strict and overflow > 0:
            raise BudgetOverflowError(report)
        for hook in self.hooks:
            hook.post_prune(report)
        return report
