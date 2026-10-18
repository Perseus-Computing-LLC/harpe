"""Dynamic Interaction Fit and Suppression Controls for Harpe.

Allows temporary damping or muting of specific memory or context clusters when dialogue context
or user steering indicates they are stale, distracting, or contradictory, without mutating permanent storage.
"""

from __future__ import annotations

import re
import time
import uuid
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from .models import ContextItem


class SuppressionScope(str, Enum):
    """Scope of entity targeting for suppression rules."""
    EXACT_ID = "exact_id"
    CATEGORY = "category"
    KEY_SUBSTRING = "key_substring"
    REGEX = "regex"
    METADATA_TAG = "metadata_tag"


class SuppressionRule(BaseModel):
    """A rule defining conditions under which context items are suppressed or damped."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    scope: SuppressionScope
    target: str = Field(description="Target pattern, category name, key substring, or tag.")
    reason: str = Field(description="Operational rationale for suppression.")
    discount_factor: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="Utility multiplier when soft-suppressed (e.g. 0.05 = 95% discount).",
    )
    hard_mute: bool = Field(
        default=False,
        description="If True, completely excludes matching items from context synthesis.",
    )
    expires_at: Optional[float] = Field(
        default=None,
        description="Unix timestamp when rule expires. None = persistent until explicitly revoked.",
    )
    created_at: float = Field(default_factory=time.time)

    def is_expired(self, now: Optional[float] = None) -> bool:
        """Check if the suppression rule has expired."""
        if self.expires_at is None:
            return False
        return (now or time.time()) > self.expires_at

    def matches(self, item: ContextItem) -> bool:
        """Evaluate if this suppression rule matches a given ContextItem."""
        if self.is_expired():
            return False

        if self.scope == SuppressionScope.EXACT_ID:
            return item.id == self.target

        elif self.scope == SuppressionScope.CATEGORY:
            item_cat = item.metadata.get("category", "")
            return str(item_cat).lower() == self.target.lower()

        elif self.scope == SuppressionScope.KEY_SUBSTRING:
            item_key = item.metadata.get("key", item.id)
            return self.target.lower() in str(item_key).lower()

        elif self.scope == SuppressionScope.REGEX:
            try:
                return bool(re.search(self.target, item.content, re.IGNORECASE))
            except re.error:
                return False

        elif self.scope == SuppressionScope.METADATA_TAG:
            tags = item.metadata.get("tags", [])
            if isinstance(tags, list):
                return self.target in tags
            return False

        return False


class SuppressionController:
    """Manages dynamic context suppression rules and applies interaction-fit damping."""

    def __init__(self):
        self._rules: Dict[str, SuppressionRule] = {}

    def add_rule(
        self,
        scope: SuppressionScope,
        target: str,
        reason: str,
        discount_factor: float = 0.05,
        hard_mute: bool = False,
        ttl_seconds: Optional[float] = None,
    ) -> SuppressionRule:
        """Register a new suppression rule."""
        expires = (time.time() + ttl_seconds) if ttl_seconds is not None else None
        rule = SuppressionRule(
            scope=scope,
            target=target,
            reason=reason,
            discount_factor=discount_factor,
            hard_mute=hard_mute,
            expires_at=expires,
        )
        self._rules[rule.id] = rule
        return rule

    def revoke_rule(self, rule_id: str) -> bool:
        """Revoke a suppression rule by id."""
        if rule_id in self._rules:
            del self._rules[rule_id]
            return True
        return False

    def list_active_rules(self) -> List[SuppressionRule]:
        """Return all active, non-expired suppression rules."""
        now = time.time()
        return [r for r in self._rules.values() if not r.is_expired(now)]

    def evaluate_item(self, item: ContextItem) -> Tuple[bool, float, Optional[str]]:
        """Evaluate an item against active rules.

        Returns:
            (is_suppressed, effective_discount_factor, suppression_reason)
        """
        active_rules = self.list_active_rules()
        for rule in active_rules:
            if rule.matches(item):
                factor = 0.0 if rule.hard_mute else rule.discount_factor
                return True, factor, rule.reason
        return False, 1.0, None

    def filter_and_damp(
        self,
        items: List[ContextItem],
    ) -> Tuple[List[ContextItem], List[ContextItem]]:
        """Separate hard-muted items and apply utility damping metadata to soft-suppressed items.

        Returns:
            (retained_items, hard_muted_items)
        """
        retained: List[ContextItem] = []
        hard_muted: List[ContextItem] = []

        for item in items:
            suppressed, factor, reason = self.evaluate_item(item)
            if suppressed:
                if factor == 0.0:
                    hard_muted.append(item)
                else:
                    # Soft damp: attach suppression metadata for knapsack utility discount
                    damped_meta = dict(item.metadata)
                    damped_meta["suppression_discount"] = factor
                    damped_meta["suppression_reason"] = reason
                    # Clone item with updated metadata
                    damped_item = item.model_copy(update={"metadata": damped_meta})
                    retained.append(damped_item)
            else:
                retained.append(item)

        return retained, hard_muted
