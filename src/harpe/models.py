"""Data models for Harpe context synthesis and token budgeting."""

from __future__ import annotations

import copy
import secrets
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field, model_validator


class PrunePolicy(str, Enum):
    """Pruning policies for context compaction."""
    FIFO = "fifo"
    LOWEST_PRIORITY = "lowest_priority"
    TRUNCATE_TAIL = "truncate_tail"
    LEAST_RELEVANT = "least_relevant"
    WARM_PREFIX = "warm_prefix"
    KNAPSACK_DP = "knapsack_dp"
    DUO_ATTENTION = "duo_attention"


class CognitiveDimension(str, Enum):
    """Cognitive activation dimensions representing distinct functional context streams."""
    INVARIANT = "invariant"    # Operational policies, trust boundaries, immutable safety constraints
    GOAL = "goal"              # Active task objectives, in-flight intent, open loops
    KNOWLEDGE = "knowledge"    # Durable declarative facts, retrieved memories
    IDENTITY = "identity"      # Agent stance, persona, self-history
    CONTEXT = "context"        # Ephemeral dialogue, execution scratchpad, tool results


class TokenBudget(BaseModel):
    """Token budget configuration and active utilization."""
    total_budget: int = Field(gt=0, description="Total window budget in tokens.")
    reserved_completion: int = Field(default=0, ge=0, description="Tokens reserved for model completion output.")
    reserved_tools: int = Field(default=0, ge=0, description="Tokens reserved for tool call definitions and overhead.")
    safety_margin: int = Field(default=0, ge=0, description="Buffer tokens reserved to prevent hard context overflow.")
    system_reserve: Optional[int] = Field(default=None, ge=0, description="Alias for safety_margin.")
    tool_reserve: Optional[int] = Field(default=None, ge=0, description="Alias for reserved_tools.")
    completion_reserve: Optional[int] = Field(default=None, ge=0, description="Alias for reserved_completion.")

    @model_validator(mode="before")
    @classmethod
    def _resolve_budget_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            data = dict(data)  # resolve on a copy: the caller's mapping is left as it was given
            # Support system_reserve -> safety_margin alias
            if "system_reserve" in data and ("safety_margin" not in data or data["safety_margin"] == 0):
                data["safety_margin"] = data["system_reserve"]
            elif "safety_margin" in data and "system_reserve" not in data:
                data["system_reserve"] = data["safety_margin"]

            # Support tool_reserve -> reserved_tools alias
            if "tool_reserve" in data and ("reserved_tools" not in data or data["reserved_tools"] == 0):
                data["reserved_tools"] = data["tool_reserve"]
            elif "reserved_tools" in data and "tool_reserve" not in data:
                data["tool_reserve"] = data["reserved_tools"]

            # Support completion_reserve -> reserved_completion alias
            if "completion_reserve" in data and ("reserved_completion" not in data or data["reserved_completion"] == 0):
                data["reserved_completion"] = data["completion_reserve"]
            elif "reserved_completion" in data and "completion_reserve" not in data:
                data["completion_reserve"] = data["reserved_completion"]
        return data

    @property
    def available_context(self) -> int:
        """Net tokens available for context items after all reservations."""
        return max(0, self.total_budget - self.reserved_completion - self.reserved_tools - self.safety_margin)

    def utilization_pct(self, used_tokens: int) -> float:
        """Percentage of available context consumed."""
        avail = self.available_context
        if avail <= 0:
            return 100.0
        return (used_tokens / avail) * 100.0

    def is_exceeded(self, used_tokens: int) -> bool:
        """Whether used tokens exceed the available context allowance."""
        return used_tokens > self.available_context


class ContextItem(BaseModel):
    """An individual context item subjected to budgeting and dynamic pruning."""
    id: str = Field(default="", description="Unique identifier for the context item.")
    item_id: Optional[str] = Field(default=None, description="Alias for id.")
    role: str = Field(
        default="context",
        pattern="^(system|user|assistant|tool|context)$",
        description="Role: system, user, assistant, tool, or context. Kept through synthesis and serialization: a "
        "tool output that contains text shaped like a user or system turn stays a tool item.",
    )
    content: str = Field(description="Raw text content of the item.")
    token_count: int = Field(default=0, ge=0, description="Calculated token length; 0 triggers counter calculation.")
    priority: int = Field(default=50, ge=0, le=100, description="Priority tier: 0 (highest/pinned) to 100 (disposable).")
    pinned: bool = Field(default=False, description="Whether this item is strictly immune to pruning.")
    created_at: float = Field(default_factory=time.time, description="Unix timestamp of item creation.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary item metadata.")
    dimension: Optional[Union[CognitiveDimension, str]] = Field(
        default=None,
        description="Cognitive activation dimension (invariant, goal, knowledge, identity, context).",
    )
    sensitivity: str = Field(
        default="normal",
        pattern="^(normal|confidential)$",
        description="'confidential' items are withheld whenever the destination is 'remote', even if pinned.",
    )

    @model_validator(mode="before")
    @classmethod
    def _resolve_item_id_alias(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Resolve on a copy: an id generated here belongs to this item only, never to the caller's dict, so a
            # later copy of that dict is a new item rather than a replay of this one's id.
            data = dict(data)
            item_id = data.get("item_id")
            cur_id = data.get("id")
            if not cur_id and item_id:
                data["id"] = str(item_id)
            elif not cur_id and not item_id:
                uid = str(uuid.uuid4())
                data["id"] = uid
                data["item_id"] = uid
            elif cur_id and not item_id:
                data["item_id"] = str(cur_id)
        return data


def _utf8(text: str) -> bytes:
    """The bytes every digest, ref and byte size in this package is taken over: UTF-8, with a lone surrogate
    written as the three bytes UTF-8 would give that code point (``errors="surrogatepass"``).

    A ``str`` may hold half of a surrogate pair (``json.loads`` of an escaped emoji cut by truncation gives one),
    and strict UTF-8 raises on it. For text without one these are exactly the strict UTF-8 bytes, so every digest
    and ref of valid text is what it was. The encoding stays one-to-one, which ``errors="replace"`` would not:
    two texts that differ only in a lone surrogate keep different bytes, so different digests and refs."""
    return text.encode("utf-8", "surrogatepass")


# Values no one can change in place: a copy shares them without looking further.
_UNCHANGEABLE = frozenset({str, int, float, bool, type(None), bytes, complex})
_CONTAINERS = (dict, list, tuple, set)


def _own_containers(value: Any, memo: Dict[int, Any]) -> Any:
    """``value`` with every ``dict``, ``list``, ``tuple`` and ``set`` in it (subclasses too) built again, down to
    any depth, and every other object shared as it is: a lock, a client, an open handle or an array is never
    cloned, so nothing here can fail on an object that cannot be copied. Writing into the result (a new key, an
    appended element) therefore never reaches ``value``; changing the state of a shared object does.

    ``memo`` maps ``id(original)`` to its copy, as in ``copy.deepcopy``: a container reachable twice is copied
    once and a container that holds itself ends. A tuple with nothing to rebuild inside is returned as it is."""
    cls = type(value)
    if cls in _UNCHANGEABLE or not isinstance(value, _CONTAINERS):
        return value
    key = id(value)
    if key in memo:
        return memo[key]
    if cls is dict:
        new_dict: Dict[Any, Any] = {}
        memo[key] = new_dict
        for k, v in value.items():
            new_dict[k] = v if type(v) in _UNCHANGEABLE else _own_containers(v, memo)
        return new_dict
    if cls is list:
        if _UNCHANGEABLE.issuperset(map(type, value)):
            memo[key] = flat = value.copy()  # an embedding, a list of tags: nothing below to own
            return flat
        new_list: List[Any] = []
        memo[key] = new_list
        new_list.extend(v if type(v) in _UNCHANGEABLE else _own_containers(v, memo) for v in value)
        return new_list
    if cls is set:
        memo[key] = new_set = set(value)  # members are hashable: shared
        return new_set
    if isinstance(value, tuple):
        if _UNCHANGEABLE.issuperset(map(type, value)):
            return value
        parts = [_own_containers(v, memo) for v in value]
        if key in memo:  # a list inside holds this tuple: it was rebuilt on the way down
            return memo[key]
        if all(new is old for new, old in zip(parts, value)):
            rebuilt = value
        else:
            try:
                rebuilt = tuple(parts) if cls is tuple else (cls(*parts) if hasattr(cls, "_fields") else cls(parts))
            except Exception:
                rebuilt = value  # a tuple subclass with a constructor of its own: left as one shared object
        memo[key] = rebuilt
        return rebuilt
    # A subclass of dict, list or set (OrderedDict, defaultdict, ...): a shallow copy keeps its type and whatever
    # else it carries, then the copy's elements are owned in place. A subclass that cannot be copied is shared
    # like any other object, and one that refuses the writes keeps the elements it has: this never raises.
    try:
        clone = copy.copy(value)
    except Exception:
        return value
    if clone is value or type(clone) is not cls:
        return value
    memo[key] = clone
    try:
        if isinstance(clone, dict):
            for k, v in list(clone.items()):
                owned = _own_containers(v, memo)
                if owned is not v:
                    clone[k] = owned
        elif isinstance(clone, list):
            for i, v in enumerate(list(clone)):
                owned = _own_containers(v, memo)
                if owned is not v:
                    clone[i] = owned
    except RecursionError:
        raise
    except Exception:
        pass
    return clone


def _detached_copy(item: ContextItem, update: Optional[Dict[str, Any]] = None) -> ContextItem:
    """A copy of ``item`` (with ``update`` applied) that shares no container with it: every field is the copy's
    own and ``metadata`` is rebuilt by :func:`_own_containers`, so a write into the copy's ``metadata``, at any
    depth, does not reach ``item`` and the other way round. Objects inside ``metadata`` that are not dicts,
    lists, tuples or sets are shared, never cloned: unlike ``model_copy(deep=True)`` this cannot raise on
    metadata that holds a lock, a client or an open handle, and it does not duplicate large payloads.

    The fields a selection, a withholding decision or a receipt reads (id, role, content, token_count, priority,
    pinned, created_at, dimension, sensitivity) are immutable values, so the copy cannot change under its owner
    through ``item``."""
    clone = item.model_copy(update=update)
    fields = clone.__dict__  # written directly, as model_copy does, so the set of explicitly set fields is kept
    if type(item) is ContextItem:
        fields["metadata"] = _own_containers(fields["metadata"], {})
        return clone
    # A subclass may add container fields (or extras) of its own: own those as well.
    memo: Dict[int, Any] = {}
    for name, value in list(fields.items()):
        fields[name] = _own_containers(value, memo)
    extra = clone.__pydantic_extra__
    if extra:
        for name, value in list(extra.items()):
            extra[name] = _own_containers(value, memo)
    return clone


class MixedRoleRenderError(ValueError):
    """Raised by ``to_context_str(mixed_roles="error")`` when the included items carry more than one role."""

    def __init__(self, roles: List[str]) -> None:
        self.roles = list(roles)
        super().__init__(f"included items carry more than one role ({', '.join(self.roles)}); a flat render would "
                         "launder them. Use to_messages(), or mixed_roles='fence' / 'flatten'.")


def _attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;").replace(">", "&gt;")


class SynthesisReport(BaseModel):
    """Synthesis result summarizing kept items, pruned items, and token accounting."""
    included_items: List[ContextItem] = Field(default_factory=list)
    pruned_items: List[ContextItem] = Field(default_factory=list)
    consumed_tokens: int = Field(default=0, ge=0)
    budget_remaining: int = Field(default=0, ge=0)
    pruned_tokens: int = Field(default=0, ge=0)
    withheld_items: List[ContextItem] = Field(
        default_factory=list,
        description="Confidential items kept out because the destination was remote. Not counted as pruned or consumed.",
    )
    policy_applied: PrunePolicy = Field(default=PrunePolicy.LOWEST_PRIORITY)
    query_intent: Optional[str] = Field(
        default=None,
        description="Classified query intent when evaluated (e.g. multi_session_aggregation, temporal, preference).",
    )
    dynamic_budget_expanded: bool = Field(
        default=False,
        description="True if the knapsack budget was dynamically expanded for multi-session aggregation.",
    )
    over_budget: bool = Field(
        default=False,
        description="True when the pinned, priority-0 and invariant items alone exceed effective_budget. They are "
        "still all included (pinned items are never pruned), so consumed_tokens is above the ceiling.",
    )
    overflow_tokens: int = Field(
        default=0,
        ge=0,
        description="How many tokens consumed_tokens is above effective_budget; 0 unless over_budget.",
    )
    control_overflow: bool = Field(
        default=False,
        description="True if invariant/pinned items alone exceeded available context (untouchable control budget overflow).",
    )
    pinned_policy_digest: str = Field(
        default="",
        description="SHA-256 digest over the pinned and invariant items in the context window (H-R1).",
    )
    knapsack_solver: Optional[str] = Field(
        default=None,
        description=(
            "Which kernel solved the exact 0/1 DP when KNAPSACK_DP or DUO_ATTENTION ran under the cell cap: "
            "'native' (harpe_native, the Rust crate) or 'python'; 'greedy' when the instance exceeded the cap; "
            "None for the other policies."
        ),
    )
    effective_budget: int = Field(
        default=0,
        ge=0,
        description="The available-context ceiling the selection was made against, after any dynamic expansion. "
        "This is the number consumed_tokens is bounded by, not the caller's TokenBudget.total_budget.",
    )
    reference_time: Optional[float] = Field(
        default=None,
        description="The timestamp recency was measured from (the caller's `now`, else the newest created_at among "
        "the items). None when there were no items. Never the wall clock, so a replay gives the same selection.",
    )
    dimension_counts: Dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown of included items by cognitive dimension.",
    )
    dimension_tokens: Dict[str, int] = Field(
        default_factory=dict,
        description="Breakdown of consumed tokens by cognitive dimension.",
    )

    def to_messages(self) -> List[Dict[str, str]]:
        """Serialize the included items role-typed, one message per item, in order.

        Nothing is merged, so a tool output that contains a forged user or system turn stays inside a tool
        message. This is the serialization a caller should hand to a chat model."""
        return [{"role": it.role, "id": it.id, "content": it.content} for it in self.included_items]

    def to_context_str(
        self,
        delimiter: str = "\n\n",
        *,
        mixed_roles: str = "fence",
        fence_nonce: Optional[str] = None,
    ) -> str:
        """Render the included items into one string.

        When every item has the same role the items are joined with ``delimiter``, as before. When roles are
        mixed, joining would launder roles (a tool output containing ``user: ...`` would read as a user turn), so
        ``mixed_roles`` decides: ``"fence"`` (default) wraps each item in a ``<harpe-role>`` block that names the
        role and carries a per-render nonce, so content produced before the render cannot forge a closing tag;
        ``"error"`` raises :class:`MixedRoleRenderError`; ``"flatten"`` is the old behaviour, on explicit opt-in
        only. ``fence_nonce`` fixes the nonce for a reproducible render (tests); otherwise it is fresh each call."""
        if mixed_roles not in ("fence", "error", "flatten"):
            raise ValueError(f"mixed_roles must be 'fence', 'error' or 'flatten', not {mixed_roles!r}")
        roles = {it.role for it in self.included_items}
        if len(roles) <= 1 or mixed_roles == "flatten":
            return delimiter.join(item.content for item in self.included_items)
        if mixed_roles == "error":
            raise MixedRoleRenderError(sorted(roles))
        nonce = fence_nonce if fence_nonce is not None else secrets.token_hex(8)
        blocks = [
            f'<harpe-role name="{it.role}" id="{_attr(it.id)}" fence="{nonce}">\n{it.content}\n</harpe-role fence="{nonce}">'
            for it in self.included_items
        ]
        return delimiter.join(blocks)

    def render(self, delimiter: str = "\n\n", **kwargs: Any) -> str:
        """Alias for to_context_str()."""
        return self.to_context_str(delimiter=delimiter, **kwargs)
