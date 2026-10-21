"""Query intent routing, fit-aware routing, dynamic knapsack budgeting, and temporal normalization for Harpe."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Iterable, Optional, Union

from pydantic import BaseModel, Field

from .models import ContextItem, TokenBudget
from .protocols import TokenCounter


class QueryIntent(str, Enum):
    """Categorical query intent for context synthesis and dynamic budgeting."""
    FACTOID_SINGLE = "factoid_single"
    MULTI_SESSION_AGGREGATION = "multi_session_aggregation"
    TEMPORAL = "temporal"
    PREFERENCE = "preference"
    GENERAL = "general"


# Intent classification regex heuristics
_AGGREGATION_PATTERN = re.compile(
    r"\b(how many times|sum|total|count|list all|every time|how often|across (all |both )?sessions|"
    r"frequency|aggregate|combine|accumulat|all occurrences|summarize all|each session)\b",
    re.IGNORECASE,
)

_TEMPORAL_PATTERN = re.compile(
    r"\b(before|after|since|during|duration|earlier|later|yesterday|tomorrow|what date|when did|"
    r"how long|first|last|timeline|chronolog|schedule|prior to|subsequent|recent|last month|"
    r"next month|last year|which year|which day)\b",
    re.IGNORECASE,
)

_PREFERENCE_PATTERN = re.compile(
    r"\b(prefer|preference|favorite|favourite|like|dislike|recommend|taste|habit|usually|"
    r"custom|would .* want|choice|tend to|fond of)\b",
    re.IGNORECASE,
)


def classify_query_intent(query: str) -> QueryIntent:
    """Classify user query into an architectural intent class for budget and knapsack routing."""
    if not query or not query.strip():
        return QueryIntent.GENERAL

    q = query.strip()

    # Multi-session aggregation takes priority for knapsack expansion
    if _AGGREGATION_PATTERN.search(q):
        return QueryIntent.MULTI_SESSION_AGGREGATION

    # Temporal queries require chronological anchors and temporal utility weighting
    if _TEMPORAL_PATTERN.search(q):
        return QueryIntent.TEMPORAL

    # Preference queries trigger dedicated persona slot injection
    if _PREFERENCE_PATTERN.search(q):
        return QueryIntent.PREFERENCE

    # Default to single-turn factoid lookup
    return QueryIntent.FACTOID_SINGLE


def is_aggregation_query(query: str) -> bool:
    """Check if query requires multi-session aggregation."""
    return bool(_AGGREGATION_PATTERN.search(query)) if query else False


def is_temporal_query(query: str) -> bool:
    """Check if query involves temporal reasoning or chronological sequence."""
    return bool(_TEMPORAL_PATTERN.search(query)) if query else False


def is_preference_query(query: str) -> bool:
    """Check if query targets implicit/explicit persona or preferences."""
    return bool(_PREFERENCE_PATTERN.search(query)) if query else False


def resolve_dynamic_budget(
    base_budget: TokenBudget,
    intent: Union[QueryIntent, str],
    aggregation_factor: float = 2.0,
    max_budget: int = 4000,
) -> TokenBudget:
    """Dynamically adjust token budget for aggregation queries while maintaining strict bounds.

    Aggregation/multi-session queries require assembling dispersed fragments across multiple sessions.
    Dynamically expanding the available budget prevents knapsack starvation without inflating single-session costs.
    """
    intent_str = str(intent.value if isinstance(intent, QueryIntent) else intent).lower()

    if intent_str in ("multi_session_aggregation", "aggregation", "multi-session"):
        expanded_total = min(int(base_budget.total_budget * aggregation_factor), max_budget)
        return base_budget.model_copy(update={"total_budget": expanded_total})

    return base_budget


def normalize_temporal_chunk(
    content: str,
    timestamp: Union[str, float, int, datetime, None] = None,
    format_str: str = "%Y-%m-%d %H:%M",
) -> str:
    """Prepend normalized ISO-8601 session timestamp to atomic turn chunks.

    Ensures relative temporal expressions ('the following day', 'three weeks prior')
    retain their chronological anchor even if intermediate turns are pruned by Harpe.
    """
    if not content:
        return ""

    # Check if chunk already starts with an explicit bracketed ISO date header, e.g. [2026-04-12]
    if re.match(r"^\[\d{4}-\d{2}-\d{2}", content.strip()):
        return content

    if timestamp is None:
        return content

    iso_str = ""
    if isinstance(timestamp, (int, float)):
        try:
            iso_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime(format_str)
        except (ValueError, OSError):
            iso_str = str(timestamp)
    elif isinstance(timestamp, datetime):
        iso_str = timestamp.strftime(format_str)
    elif isinstance(timestamp, str):
        iso_str = timestamp.strip()

    if iso_str:
        return f"[{iso_str}] {content}"

    return content


def has_temporal_anchor(content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
    """Check if a context item contains explicit temporal anchor markers."""
    if metadata:
        if any(k in metadata for k in ("temporal_anchor", "timestamp", "session_date", "date")):
            return True
    if content and re.match(r"^\[\d{4}-\d{2}-\d{2}", content.strip()):
        return True
    return False


def create_persona_context_item(
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    token_count: Optional[int] = None,
) -> ContextItem:
    """Construct a dedicated pinned persona slot for Harpe's WARM_PREFIX region.

    Injected at priority=0 and pinned=True to guarantee retention at zero knapsack eviction risk.
    """
    meta = metadata.copy() if metadata else {}
    meta["is_persona_slot"] = True
    meta["category"] = "persona"

    kwargs: Dict[str, Any] = {
        "content": content,
        "role": "system",
        "priority": 0,
        "pinned": True,
        "metadata": meta,
    }
    if token_count is not None:
        kwargs["token_count"] = token_count

    return ContextItem(**kwargs)


class ContextRoute(str, Enum):
    """How to build the context for a question over one history."""
    WHOLE = "whole"  # the history fits under the threshold: hand over every item, in order, unpruned
    RETRIEVE = "retrieve"  # it does not (or routing is off): retrieve candidates and synthesize within the budget


class RouteDecision(BaseModel):
    """What :func:`route_by_fit` decided, and the tokens it counted."""
    route: ContextRoute
    threshold_tokens: int = Field(default=0, ge=0, description="The threshold applied; 0 when routing is off.")
    whole_tokens: Optional[int] = Field(
        default=None, ge=0, description="Tokens of the whole history when it fits; None when it does not."
    )


def route_by_fit(
    items: Iterable[Union[ContextItem, str]],
    threshold_tokens: Optional[int],
    token_counter: Optional[TokenCounter] = None,
) -> RouteDecision:
    """Fit-aware routing: hand a whole history over when it fits under ``threshold_tokens``, else retrieve.

    A history that fits the reader's window whole needs no retrieval: passing every item, in its order,
    gives the reader everything at a cost bounded by the threshold, and nothing can be missed. Above the
    threshold the caller retrieves and synthesizes within its budget as usual. ``items`` are the history's
    items (strings or :class:`ContextItem`, whose ``token_count`` is used when set); counting stops at the
    first item that takes the total over the threshold. ``threshold_tokens`` of ``None`` or ``0`` turns
    routing off (always ``RETRIEVE``). Count in the reader's tokens, and pick a threshold the reader's
    window and price can take: telarch-benchmarks measured 32,000 tokens.
    """
    if not threshold_tokens or threshold_tokens <= 0:
        return RouteDecision(route=ContextRoute.RETRIEVE, threshold_tokens=0)
    if token_counter is None:
        from .engine import HeuristicTokenCounter  # engine imports this module

        token_counter = HeuristicTokenCounter()
    total = 0
    for item in items:
        if isinstance(item, ContextItem):
            total += item.token_count or token_counter.count_tokens(item.content)
        else:
            total += token_counter.count_tokens(item)
        if total > threshold_tokens:
            return RouteDecision(route=ContextRoute.RETRIEVE, threshold_tokens=threshold_tokens)
    return RouteDecision(route=ContextRoute.WHOLE, threshold_tokens=threshold_tokens, whole_tokens=total)
