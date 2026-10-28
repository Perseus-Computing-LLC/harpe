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
    """Evaluates candidate actions against active operational invariants and epistemic boundaries.

    The blocked patterns are a last-resort heuristic over the action string, not a sandbox: they catch the
    obvious spellings of a few destructive commands (recursive ``rm`` of root, home, a glob or the parent
    directory, with the flags in any order or split across groups; ``chmod 777``; ``curl ... | sh``; exporting a
    token or key) and nothing more. Quoting, aliases, variables, scripts and every command they do not name pass
    straight through. Enforce real limits in the executor, and treat a REJECTED verdict here as a tripwire that
    fired, not as proof that APPROVED actions are safe."""

    # `rm` followed by option groups, at least one of which carries `r` or `R` (recursive), with any number of
    # other groups around it (`-rf`, `-fr`, `-r -f`, `--recursive --force`), then the target.
    _RM_RECURSIVE = (
        r"\brm\s+(?:(?:-[a-zA-Z]+|--[a-z-]+)\s+)*(?:-[a-zA-Z]*[rR][a-zA-Z]*|--recursive)\s+(?:(?:-[a-zA-Z]+|--[a-z-]+)\s+)*"
    )
    DEFAULT_BLOCKED_PATTERNS: List[str] = [
        # Targets: anything under `/`, `~` or `$HOME`; any argument carrying a `*` glob; `.` or `..` alone.
        _RM_RECURSIVE + r"(?:[/~]|\$HOME|\S*\*|\.{1,2}/?(?:\s|$))",
        r"\bchmod\s+(?:-[a-zA-Z]+\s+)*0?777\b",
        r"\b(?:curl|wget)\b.*\|\s*(?:sudo\s+)?(?:ba|z|da)?sh\b",
        r"\bexport\s+\S*TOKEN\S*=",
        r"\bexport\s+\S*KEY\S*=",
    ]

    def __init__(self, blocked_patterns: Optional[List[str]] = None):
        self.blocked_patterns = (
            list(blocked_patterns) if blocked_patterns is not None else list(self.DEFAULT_BLOCKED_PATTERNS)
        )

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

        # 1. Check intrinsic blocked patterns. Whitespace is collapsed first so tabs, newlines and doubled
        # spaces cannot carry a flag group past the patterns.
        normalised_action = " ".join(candidate_action.split())
        for pattern in self.blocked_patterns:
            if re.search(pattern, normalised_action, re.IGNORECASE):
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

            # Rule: Epistemic boundary discipline. Only a non-empty key can match: '' is a substring of every
            # string, so a revoked invariant without a key used to suppress every action.
            if "revoked" in str(inv.metadata.get("epistemic_state", "")).lower():
                key = str(inv.metadata.get("key", "") or "").strip().lower()
                if key and key in candidate_action.lower():
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
