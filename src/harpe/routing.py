"""Query intent routing, fit-aware routing, dynamic knapsack budgeting, and temporal normalization for Harpe."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
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


_AGGREGATION_INTENTS = frozenset({"multi_session_aggregation", "aggregation", "multi-session"})


def resolve_dynamic_budget(
    base_budget: TokenBudget,
    intent: Union[QueryIntent, str],
    aggregation_factor: float = 2.0,
    max_factor: float = 2.0,
) -> TokenBudget:
    """Widen ``total_budget`` for aggregation queries by ``min(aggregation_factor, max_factor)``, never below 1.0,
    so the result is never smaller than the caller's budget. Other intents return ``base_budget`` unchanged.

    Aggregation/multi-session queries assemble fragments spread across sessions, and a budget sized for one
    session starves the knapsack. The cap is a factor of the caller's budget rather than an absolute token count:
    an absolute cap (formerly 4000) silently halved any budget already above it while the report said "expanded".
    The reserves are copied as they are; only ``total_budget`` grows.
    """
    intent_str = str(intent.value if isinstance(intent, QueryIntent) else intent).lower()
    if intent_str not in _AGGREGATION_INTENTS:
        return base_budget
    # >= 1.0: expansion can only grow the budget, whatever the caller passed for the factors.
    factor = max(1.0, min(aggregation_factor, max_factor))
    return base_budget.model_copy(update={"total_budget": int(base_budget.total_budget * factor)})


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


_MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
           "november", "december"]
_MONTH_NUM = {**{m: i + 1 for i, m in enumerate(_MONTHS)}, **{m[:3]: i + 1 for i, m in enumerate(_MONTHS)}, "sept": 9}
_MONTH_RE = r"(" + "|".join(sorted(_MONTH_NUM, key=len, reverse=True)) + r")\.?"
_ANCHOR_PATTERNS = (
    (re.compile(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})"), ("y", "m", "d")),
    (re.compile(r"(\d{1,2})(?:st|nd|rd|th)?\s+" + _MONTH_RE + r",?\s+(\d{4})", re.IGNORECASE), ("d", "mn", "y")),
    (re.compile(_MONTH_RE + r"[\s-]+(\d{1,2})(?:st|nd|rd|th)?,?[\s-]+(\d{4})", re.IGNORECASE), ("mn", "d", "y")),
)
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
_NUMBER_WORDS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
                 "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "a couple of": 2, "couple of": 2}
_NUM = r"(?:\d{1,3}|a couple of|couple of|" + "|".join(sorted(_NUMBER_WORDS, key=len, reverse=True)) + r")"
_UNIT = r"(?:days?|weeks?|months?|years?)"
_RELATIVE = re.compile(
    r"\b(?:"
    r"(?P<dby>the day before yesterday)|(?P<dat>the day after tomorrow)|(?P<yd>yesterday)|(?P<ln>last night)"
    r"|(?P<tm>tomorrow)"
    rf"|(?P<ago>(?P<ago_n>{_NUM})\s+(?P<ago_u>{_UNIT})\s+ago)"
    rf"|(?P<fn>(?P<fn_n>{_NUM})\s+(?P<fn_u>{_UNIT})\s+from now)"
    r"|(?P<we>(?P<we_r>last|next|this)\s+weekend)"
    r"|(?P<wk>(?P<wk_r>last|next|this)\s+week)"
    r"|(?P<mo>(?P<mo_r>last|next|this)\s+month)"
    r"|(?P<yr>(?P<yr_r>last|next|this)\s+year)"
    r"|(?P<wd>last\s+(?P<wd_d>" + "|".join(_WEEKDAYS) + r"))"
    r")\b",
    re.IGNORECASE,
)


def parse_anchor_date(value: Union[str, date, datetime, None]) -> Optional[date]:
    """The calendar date in a session timestamp such as ``1:56 pm on 8 May, 2023``, ``2023/05/20 (Sat) 02:21``,
    ``March-15-2024`` or ``2024-03-15``; ``None`` when there is none."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not value:
        return None
    for pattern, order in _ANCHOR_PATTERNS:
        m = pattern.search(str(value))
        if not m:
            continue
        parts = dict(zip(order, m.groups()))
        month = int(parts["m"]) if "m" in parts else _MONTH_NUM.get(parts["mn"].lower().rstrip("."))
        try:
            return date(int(parts["y"]), int(month or 0), int(parts["d"]))
        except (TypeError, ValueError):
            continue
    return None


def _day(d: date) -> str:
    return f"{d.day} {_MONTHS[d.month - 1].capitalize()} {d.year}"


def _span(a: date, b: date) -> str:
    if (a.year, a.month) == (b.year, b.month):
        return f"{a.day}-{b.day} {_MONTHS[a.month - 1].capitalize()} {a.year}"
    if a.year == b.year:
        return f"{a.day} {_MONTHS[a.month - 1].capitalize()} - {_day(b)}"
    return f"{_day(a)} - {_day(b)}"


def _shift_months(d: date, months: int) -> date:
    y, m = divmod(d.year * 12 + d.month - 1 + months, 12)
    return date(y, m + 1, 1)


def _count(word: str) -> int:
    word = " ".join(word.lower().split())
    return int(word) if word.isdigit() else _NUMBER_WORDS[word]


def _offset(anchor: date, n: int, unit: str) -> str:
    unit = unit.lower().rstrip("s")
    if unit == "day":
        return _day(anchor + timedelta(days=n))
    if unit == "week":
        return _day(anchor + timedelta(weeks=n))
    if unit == "month":
        d = _shift_months(anchor, n)
        return f"{_MONTHS[d.month - 1].capitalize()} {d.year}"
    return str(anchor.year + n)


def _resolve(m: "re.Match[str]", anchor: date) -> Optional[str]:
    g = m.group
    if g("dby"):
        return _day(anchor - timedelta(days=2))
    if g("dat"):
        return _day(anchor + timedelta(days=2))
    if g("yd") or g("ln"):
        return _day(anchor - timedelta(days=1))
    if g("tm"):
        return _day(anchor + timedelta(days=1))
    if g("ago"):
        return _offset(anchor, -_count(g("ago_n")), g("ago_u"))
    if g("fn"):
        return _offset(anchor, _count(g("fn_n")), g("fn_u"))
    if g("we"):
        saturday = anchor + timedelta(days=5 - anchor.weekday())  # this week's Saturday (weeks start on Monday)
        rel = g("we_r").lower()
        if rel == "last":
            saturday -= timedelta(days=7)
        elif rel == "next":
            saturday += timedelta(days=7)
        return _span(saturday, saturday + timedelta(days=1))
    if g("wk"):
        monday = anchor - timedelta(days=anchor.weekday())
        monday += timedelta(days={"last": -7, "next": 7}.get(g("wk_r").lower(), 0))
        return "week of " + _day(monday)
    if g("mo"):
        d = _shift_months(anchor, {"last": -1, "next": 1}.get(g("mo_r").lower(), 0))
        return f"{_MONTHS[d.month - 1].capitalize()} {d.year}"
    if g("yr"):
        return str(anchor.year + {"last": -1, "next": 1}.get(g("yr_r").lower(), 0))
    if g("wd"):
        back = (anchor.weekday() - _WEEKDAYS.index(g("wd_d").lower())) % 7 or 7
        return _day(anchor - timedelta(days=back))
    return None


def resolve_relative_dates(text: str, anchor: Union[str, date, datetime, None]) -> str:
    """Add the calendar date to each relative time expression in ``text``, against the date it was said on:
    ``yesterday`` said on 11 May 2023 becomes ``yesterday [10 May 2023]``, ``last weekend`` gains its two days,
    ``three weeks ago`` its date, ``last month`` its month.

    A reader that sees a remembered turn out of its conversation (or after pruning) can then place it in time
    without doing the date arithmetic itself. Expressions whose meaning is ambiguous (``in two weeks``, ``next
    Friday``) are left alone, as is text whose ``anchor`` has no date (:func:`parse_anchor_date`). Expressions
    already followed by a bracketed date are skipped, so resolving twice changes nothing more.
    """
    day = parse_anchor_date(anchor)
    if day is None or not text:
        return text

    def annotate(m: "re.Match[str]") -> str:
        if text[m.end():m.end() + 2] == " [":
            return m.group(0)
        resolved = _resolve(m, day)
        return f"{m.group(0)} [{resolved}]" if resolved else m.group(0)

    return _RELATIVE.sub(annotate, text)
