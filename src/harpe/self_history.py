"""Agent Self-History and Reflective Continuity for Harpe.

Tracks the agent's own past decisions, tool invocations, and outcomes separately
from general user or domain facts, providing reflective continuity across sessions.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .models import CognitiveDimension, ContextItem


class ActionOutcome(str, Enum):
    """Execution outcome of a past agent action or tool invocation."""
    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    ABORTED = "aborted"


class SelfHistoryRecord(BaseModel):
    """A record of an agent action, tool invocation, and retrospective reflection."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    action_type: str = Field(description="Type of action (e.g. tool_call, edit, terminal, git_push).")
    description: str = Field(description="Description of what was executed.")
    outcome: ActionOutcome = Field(default=ActionOutcome.SUCCESS)
    lessons_learned: str = Field(default="", description="Retrospective learning or correction.")
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentSelfHistoryTracker:
    """Manages the agent's reflective history and exports it as identity context."""

    def __init__(self, max_records: int = 50):
        self._records: List[SelfHistoryRecord] = []
        self._max_records = max_records

    def record_action(
        self,
        action_type: str,
        description: str,
        outcome: ActionOutcome = ActionOutcome.SUCCESS,
        lessons_learned: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SelfHistoryRecord:
        """Record an executed action and optional reflection."""
        rec = SelfHistoryRecord(
            action_type=action_type,
            description=description,
            outcome=outcome,
            lessons_learned=lessons_learned,
            metadata=metadata or {},
        )
        self._records.append(rec)
        if len(self._records) > self._max_records:
            self._records.pop(0)
        return rec

    def get_recent(
        self,
        limit: int = 5,
        outcome_filter: Optional[ActionOutcome] = None,
    ) -> List[SelfHistoryRecord]:
        """Return the most recent self-history records, newest first."""
        records = reversed(self._records)
        if outcome_filter is not None:
            records = (r for r in records if r.outcome == outcome_filter)
        result: List[SelfHistoryRecord] = []
        for r in records:
            result.append(r)
            if len(result) >= limit:
                break
        return result

    def to_context_items(self, limit: int = 3) -> List[ContextItem]:
        """Render recent self-history records into ContextItem objects under CognitiveDimension.IDENTITY."""
        items: List[ContextItem] = []
        recent = self.get_recent(limit=limit)
        for r in recent:
            lesson_part = f" | Lesson: {r.lessons_learned}" if r.lessons_learned else ""
            content = f"[Agent Self-History] {r.action_type}: {r.description} -> {r.outcome.value.upper()}{lesson_part}"
            items.append(
                ContextItem(
                    id=f"self_history_{r.id}",
                    role="system",
                    content=content,
                    priority=20,  # High priority identity context
                    dimension=CognitiveDimension.IDENTITY,
                    metadata={"action_type": r.action_type, "outcome": r.outcome.value, "record_id": r.id},
                )
            )
        return items
