"""Tests for H-R1: Preflight-reserve invariant block + system policy as untouchable control budget.

H-R1 Acceptance:
- Preflight-reserve invariant block + system policy as untouchable control budget.
- Fail closed with Argus control_overflow when control budget overflows.
- pinned_policy_digest stamped on synthesis reports and events.
"""

import hashlib
import pytest

from harpe import (
    CognitiveDimension,
    ContextBudgetEngine,
    ContextItem,
    TokenBudget,
)
from harpe.engine import BudgetOverflowError, ControlOverflowError


def _invariant_item(text: str, id_: str = "inv"):
    return ContextItem(
        id=id_,
        role="system",
        content=text,
        dimension=CognitiveDimension.INVARIANT,
        pinned=True,
        priority=0,
    )


def test_pinned_policy_digest_present_and_deterministic():
    """H-R1: pinned_policy_digest is deterministic and reflects invariant content."""
    engine = ContextBudgetEngine()
    budget = TokenBudget(total_budget=1000)

    item1 = _invariant_item("NIST SP 800-53 AC-3 Access Control Enforced", "inv_1")
    item2 = _invariant_item("Do not disclose API credentials", "inv_2")
    user_item = ContextItem(id="usr_1", role="user", content="Hello world", priority=50)

    report = engine.synthesize([item1, item2, user_item], budget)
    assert report.pinned_policy_digest != ""
    assert report.control_overflow is False

    # Determinism: reordering input items produces the exact same policy digest
    report_reordered = engine.synthesize([user_item, item2, item1], budget)
    assert report_reordered.pinned_policy_digest == report.pinned_policy_digest

    # Changing policy content changes digest
    item2_mod = _invariant_item("Do not disclose API credentials or tokens", "inv_2")
    report_mod = engine.synthesize([item1, item2_mod, user_item], budget)
    assert report_mod.pinned_policy_digest != report.pinned_policy_digest


def test_control_overflow_fails_closed_in_strict_mode():
    """H-R1: when invariant policy alone exceeds budget, fail closed with ControlOverflowError."""
    engine = ContextBudgetEngine()
    # A tiny budget where invariants cannot fit
    budget = TokenBudget(total_budget=30)

    large_policy = _invariant_item(
        "Mandatory operational invariant: All outbound network sockets must be TLS 1.3 encrypted. "
        "All credentials must be validated against hardware HSM. Audit logging must run synchronously. "
        "No unauthenticated shell execution allowed under any circumstance.",
        "inv_giant",
    )
    user_item = ContextItem(id="usr_1", role="user", content="run tool", priority=50)

    # In strict mode, must raise ControlOverflowError
    with pytest.raises(ControlOverflowError) as exc_info:
        engine.synthesize([large_policy, user_item], budget, strict=True)

    err = exc_info.value
    assert isinstance(err, BudgetOverflowError)
    assert err.report.control_overflow is True
    assert err.report.over_budget is True
    assert err.report.overflow_tokens > 0
    assert err.report.pinned_policy_digest != ""


def test_control_overflow_flag_set_in_non_strict_mode():
    """H-R1: in non-strict mode, report records control_overflow=True."""
    engine = ContextBudgetEngine()
    budget = TokenBudget(total_budget=30)

    large_policy = _invariant_item(
        "Mandatory operational invariant: All outbound network sockets must be TLS 1.3 encrypted. "
        "All credentials must be validated against hardware HSM. Audit logging must run synchronously.",
        "inv_giant",
    )
    report = engine.synthesize([large_policy], budget, strict=False)
    assert report.control_overflow is True
    assert report.over_budget is True
    assert report.overflow_tokens > 0
    assert report.pinned_policy_digest != ""
