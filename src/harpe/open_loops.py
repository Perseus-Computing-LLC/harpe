"""Open Loops tracking for Harpe context budgeting.

Maintains multi-turn commitments, pending agent obligations, and partially completed tasks,
ensuring continuity across dialogue turns without loss during context eviction.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .models import CognitiveDimension, ContextItem


class OpenLoopStatus(str, Enum):
    """Lifecycle states of an open obligation or commitment."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    DROPPED = "dropped"


class OpenLoop(BaseModel):
    """A tracked goal, commitment, or multi-turn obligation."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    title: str = Field(description="Short human-readable title of the obligation.")
    description: str = Field(default="", description="Detailed spec or criteria for fulfillment.")
    status: OpenLoopStatus = Field(default=OpenLoopStatus.OPEN)
    priority: int = Field(default=15, ge=0, le=100, description="Priority (0 highest, 100 lowest). Defaults to high (15).")
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)
    resolution_summary: Optional[str] = Field(default=None, description="Summary of how the loop was closed.")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        """True if the loop is unresolved and active."""
        return self.status in (OpenLoopStatus.OPEN, OpenLoopStatus.IN_PROGRESS)


class OpenLoopTracker:
    """Manages active agent obligations and renders them as prioritized cognitive goals."""

    def __init__(self):
        self._loops: Dict[str, OpenLoop] = {}

    def register(
        self,
        title: str,
        description: str = "",
        priority: int = 15,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OpenLoop:
        """Register a new open loop commitment."""
        loop = OpenLoop(
            title=title,
            description=description,
            priority=priority,
            metadata=metadata or {},
        )
        self._loops[loop.id] = loop
        return loop

    def update_status(
        self,
        loop_id: str,
        status: OpenLoopStatus,
        summary: Optional[str] = None,
    ) -> OpenLoop:
        """Update the lifecycle status of an open loop."""
        if loop_id not in self._loops:
            raise KeyError(f"OpenLoop '{loop_id}' not found.")
        loop = self._loops[loop_id]
        loop.status = status
        loop.updated_at = time.time()
        if summary is not None:
            loop.resolution_summary = summary
        return loop

    def resolve(self, loop_id: str, resolution_summary: str = "") -> OpenLoop:
        """Mark an open loop as successfully resolved."""
        return self.update_status(loop_id, OpenLoopStatus.RESOLVED, summary=resolution_summary)

    def drop(self, loop_id: str, reason: str = "") -> OpenLoop:
        """Drop an open loop without completion (abandoned or superseded)."""
        return self.update_status(loop_id, OpenLoopStatus.DROPPED, summary=reason)

    def get(self, loop_id: str) -> Optional[OpenLoop]:
        """Fetch an open loop by id."""
        return self._loops.get(loop_id)

    def list_active(self) -> List[OpenLoop]:
        """Return all active (open or in_progress) loops ordered by priority."""
        active = [loop for loop in self._loops.values() if loop.is_active]
        return sorted(active, key=lambda l: (l.priority, l.created_at))

    def list_all(self) -> List[OpenLoop]:
        """Return all tracked loops."""
        return list(self._loops.values())

    def to_context_items(self) -> List[ContextItem]:
        """Convert active open loops into prioritized ContextItem instances with CognitiveDimension.GOAL."""
        items: List[ContextItem] = []
        for loop in self.list_active():
            desc_part = f": {loop.description}" if loop.description else ""
            content = f"[Active Open Loop #{loop.id[:8]}] {loop.title}{desc_part} (Status: {loop.status.value})"
            items.append(
                ContextItem(
                    id=f"open_loop_{loop.id}",
                    role="system",
                    content=content,
                    priority=loop.priority,
                    dimension=CognitiveDimension.GOAL,
                    metadata={"loop_id": loop.id, "status": loop.status.value, **loop.metadata},
                )
            )
        return items
