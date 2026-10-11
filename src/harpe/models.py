"""Data models for Harpe context synthesis and token budgeting."""

from __future__ import annotations

import time
from enum import Enum
from typing import Any, Dict, List
from pydantic import BaseModel, Field


class PrunePolicy(str, Enum):
    """Pruning policies for context compaction."""
    FIFO = "fifo"
    LOWEST_PRIORITY = "lowest_priority"
    TRUNCATE_TAIL = "truncate_tail"
    LEAST_RELEVANT = "least_relevant"


class TokenBudget(BaseModel):
    """Token budget configuration and active utilization."""
    total_budget: int = Field(gt=0, description="Total window budget in tokens.")
    reserved_completion: int = Field(default=0, ge=0, description="Tokens reserved for model completion output.")
    reserved_tools: int = Field(default=0, ge=0, description="Tokens reserved for tool call definitions and overhead.")
    safety_margin: int = Field(default=0, ge=0, description="Buffer tokens reserved to prevent hard context overflow.")

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
    id: str = Field(description="Unique identifier for the context item.")
    role: str = Field(default="context", description="Role: system, user, assistant, tool, or context.")
    content: str = Field(description="Raw text content of the item.")
    token_count: int = Field(default=0, ge=0, description="Calculated token length; 0 triggers counter calculation.")
    priority: int = Field(default=50, ge=0, le=100, description="Priority tier: 0 (highest/pinned) to 100 (disposable).")
    pinned: bool = Field(default=False, description="Whether this item is strictly immune to pruning.")
    created_at: float = Field(default_factory=time.time, description="Unix timestamp of item creation.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary item metadata.")
    sensitivity: str = Field(
        default="normal",
        pattern="^(normal|confidential)$",
        description="'confidential' items are withheld whenever the destination is 'remote', even if pinned.",
    )


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
