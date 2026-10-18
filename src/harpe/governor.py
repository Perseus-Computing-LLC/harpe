"""Behavioral Selection Governor for Harpe context and agent execution.

Enforces Governed Retained-State Behavioral Selection:
'Retained history may influence future behaviour, but it does not automatically control it.'

Verifies that proposed agent actions adhere to active operational invariants,
screens out contradictory or revoked historical directives, and gates execution.
"""

from __future__ import annotations

import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from .models import CognitiveDimension, ContextItem


class GovernorVerdict(str, Enum):
    """The gating verdict reached by the Governor."""
    APPROVED = "approved"
    SUPPRESSED = "suppressed"
    NEEDS_REVIEW = "needs_review"
    REJECTED = "rejected"


class GovernorDecision(BaseModel):
    """Structured audit trail of a Governor evaluation."""
    verdict: GovernorVerdict
    action: str
    reasons: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    evaluated_invariants: int = 0
    confidence: float = 1.0
    timestamp: float = Field(default_factory=time.time)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BehavioralGovernor:
    """Evaluates candidate actions against active operational invariants and epistemic boundaries."""

    def __init__(self, blocked_patterns: Optional[List[str]] = None):
        # Default safety patterns that trigger scrutiny unless overridden
        self.blocked_patterns = blocked_patterns or [
            r"rm\s+-rf\s+[/~]",
            r"chmod\s+777",
            r"curl.*\|\s*sh",
            r"export\s+.*TOKEN=.*",
            r"export\s+.*KEY=.*",
        ]

    def evaluate(
        self,
        candidate_action: str,
        active_invariants: List[ContextItem],
        candidate_history: Optional[List[ContextItem]] = None,
        require_all_invariants_satisfied: bool = True,
    ) -> GovernorDecision:
        """Evaluate a candidate action against operational invariants and historical context.

        Args:
            candidate_action: The command, API call, or proposed action string.
            active_invariants: Context items with dimension == CognitiveDimension.INVARIANT or pinned=True.
            candidate_history: Optional historical context items to check for contradictory premises.
            require_all_invariants_satisfied: If True, any direct violation causes rejection.

        Returns:
            GovernorDecision with verdict and rationale.
        """
        reasons: List[str] = []
        contradictions: List[str] = []

        # 1. Check intrinsic blocked patterns
        for pattern in self.blocked_patterns:
            if re.search(pattern, candidate_action, re.IGNORECASE):
                reasons.append(f"Action matches critical blocked pattern: '{pattern}'")
                return GovernorDecision(
                    verdict=GovernorVerdict.REJECTED,
                    action=candidate_action,
                    reasons=reasons,
                    contradictions=contradictions,
                    evaluated_invariants=len(active_invariants),
                )

        # 2. Evaluate against active operational invariants
        invariants_checked = 0
        for inv in active_invariants:
            invariants_checked += 1
            inv_text = inv.content.lower()

            # Rule: Plaintext credential leakage detection
            if "never store secret values" in inv_text or "credentials out of conversation" in inv_text:
                if any(k in candidate_action.lower() for k in ["api_key=", "secret=", "password=", "bearer "]):
                    reasons.append(f"Violates credential privacy invariant: '{inv.id}'")
                    return GovernorDecision(
                        verdict=GovernorVerdict.REJECTED,
                        action=candidate_action,
                        reasons=reasons,
                        contradictions=contradictions,
                        evaluated_invariants=invariants_checked,
                    )

            # Rule: Clean room and licensing adherence
            if "mit license" in inv_text and "proprietary" in candidate_action.lower():
                contradictions.append(f"Contradicts invariant '{inv.id}': Action references proprietary licensing under MIT requirement.")

            # Rule: Epistemic boundary discipline
            if "revoked" in inv.metadata.get("epistemic_state", "").lower():
                if inv.metadata.get("key", "").lower() in candidate_action.lower():
                    reasons.append(f"Action relies on revoked epistemic state in '{inv.id}'.")

        # 3. Check for contradictory directives in candidate history
        if candidate_history:
            for hist in candidate_history:
                hist_state = hist.metadata.get("epistemic_state", "")
                if hist_state == "revoked":
                    contradictions.append(f"Candidate history item '{hist.id}' has been revoked.")

        # Determine verdict based on findings
        if contradictions and require_all_invariants_satisfied:
            return GovernorDecision(
                verdict=GovernorVerdict.NEEDS_REVIEW,
                action=candidate_action,
                reasons=reasons or ["Historical context contains unresolved contradictions."],
                contradictions=contradictions,
                evaluated_invariants=invariants_checked,
            )

        if reasons:
            return GovernorDecision(
                verdict=GovernorVerdict.SUPPRESSED,
                action=candidate_action,
                reasons=reasons,
                contradictions=contradictions,
                evaluated_invariants=invariants_checked,
            )

        return GovernorDecision(
            verdict=GovernorVerdict.APPROVED,
            action=candidate_action,
            reasons=["All operational invariants satisfied."],
            contradictions=[],
            evaluated_invariants=invariants_checked,
        )
