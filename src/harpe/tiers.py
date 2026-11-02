"""ICM tiers read off the fields a ``ContextItem`` already has (Telarch master roadmap 17.2, phase P14.1).

The five tiers are a reading of ``pinned``, ``priority`` and ``dimension``, not new state:

====  ===============  ==========================================================================================
tier  name             items
====  ===============  ==========================================================================================
0     CONTROL          pinned, priority-0 or invariant-dimension items: the control budget, which
                       ``ContextBudgetEngine`` never prunes (exactly ``ContextBudgetEngine._is_never_pruned``)
1     WORKSPACE_MAP    priority 1-19: the workspace map
2     TASK             priority 20-39: the task
3     INDEX            priority 40-69: frontmatter index rows (title + description); the default priority, 50
4     BODY             priority 70-100: bodies, loaded on demand
====  ===============  ==========================================================================================

Nothing here changes selection. ``tier_item`` sets only ``pinned`` and ``priority`` on a plain ``ContextItem``,
so every prune policy treats a tier-built item exactly as it treats a hand-built item with the same fields, and
``SynthesisReport.pinned_policy_digest`` (role, dimension, id, content of the tier-0 items) is untouched.
Under ``LOWEST_PRIORITY`` the bands keep lower tiers first through the existing priority order; ``strict=True``
still raises ``ControlOverflowError`` when the tier-0 items alone exceed the capacity.
"""

from __future__ import annotations

from enum import IntEnum
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union

from .engine import ContextBudgetEngine
from .models import ContextItem

__all__ = ["IcmTier", "TIER_PRIORITY_BANDS", "icm_tier", "tier_breakdown", "tier_item"]


class IcmTier(IntEnum):
    """The five ICM tiers; the value is the tier number."""

    CONTROL = 0
    WORKSPACE_MAP = 1
    TASK = 2
    INDEX = 3
    BODY = 4


# Inclusive priority band per tier. Together they partition ContextItem's priority range, 0..100. Tier 0's band
# is the priority-0 way in; pinned and invariant-dimension items are tier 0 at any priority.
TIER_PRIORITY_BANDS: Mapping[IcmTier, Tuple[int, int]] = MappingProxyType(
    {
        IcmTier.CONTROL: (0, 0),
        IcmTier.WORKSPACE_MAP: (1, 19),
        IcmTier.TASK: (20, 39),
        IcmTier.INDEX: (40, 69),
        IcmTier.BODY: (70, 100),
    }
)


def icm_tier(item: ContextItem) -> IcmTier:
    """The tier ``item`` reads as. Tier 0 is the engine's own never-pruned test, so the two cannot drift."""
    if ContextBudgetEngine._is_never_pruned(item):
        return IcmTier.CONTROL
    for tier in (IcmTier.WORKSPACE_MAP, IcmTier.TASK, IcmTier.INDEX, IcmTier.BODY):
        lo, hi = TIER_PRIORITY_BANDS[tier]
        if lo <= item.priority <= hi:
            return tier
    # ContextItem validates priority to 0..100 and the bands cover 1..100, so only a bypassed validator lands here.
    raise ValueError(f"priority {item.priority!r} is outside every ICM tier band")


def tier_item(
    tier: Union[IcmTier, int],
    content: str,
    *,
    priority: Optional[int] = None,
    **fields: Any,
) -> ContextItem:
    """A plain ``ContextItem`` in ``tier``: tier 0 is pinned at priority 0, tiers 1-4 are unpinned at the bottom
    of their band unless ``priority`` names another value inside it. Every other field (id, role, dimension,
    metadata, token_count, ...) passes through unchanged.

    Raises ``ValueError`` for an unknown tier, a priority outside the tier's band, or fields that would read as
    another tier (an invariant-dimension item is tier 0); ``TypeError`` when ``pinned`` is passed, since the tier
    decides it.
    """
    tier = IcmTier(tier)
    if "pinned" in fields:
        raise TypeError("tier_item sets pinned from the tier; pass the tier, not pinned")
    lo, hi = TIER_PRIORITY_BANDS[tier]
    if priority is None:
        priority = lo
    elif not lo <= priority <= hi:
        raise ValueError(f"priority {priority} is outside tier {int(tier)} ({tier.name}) band {lo}..{hi}")
    item = ContextItem(content=content, pinned=tier is IcmTier.CONTROL, priority=priority, **fields)
    read_as = icm_tier(item)
    if read_as is not tier:
        raise ValueError(
            f"these fields read as tier {int(read_as)} ({read_as.name}), not tier {int(tier)} ({tier.name})"
        )
    return item


def tier_breakdown(items: Iterable[ContextItem]) -> Tuple[Dict[int, int], Dict[int, int]]:
    """``(counts, tokens)`` per tier, keyed by tier number 0..4 in order, every tier present (zero when empty).
    Plain int keys and values, so the result can go into a receipt as it is."""
    counts: Dict[int, int] = {int(t): 0 for t in IcmTier}
    tokens: Dict[int, int] = {int(t): 0 for t in IcmTier}
    for item in items:
        t = int(icm_tier(item))
        counts[t] += 1
        tokens[t] += int(item.token_count)
    return counts, tokens
