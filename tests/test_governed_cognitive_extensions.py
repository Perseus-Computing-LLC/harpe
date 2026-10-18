"""Tests for Harpe Governed Cognitive Extensions (CAAI / Evolution 2 Integration).

Tests Open Loops, Dynamic Suppression Controls, Behavioral Selection Governor, and Agent Self-History.
"""

import time
import pytest

from harpe import (
    ActionOutcome,
    AgentSelfHistoryTracker,
    BehavioralGovernor,
    CognitiveDimension,
    ContextItem,
    GovernorVerdict,
    OpenLoopStatus,
    OpenLoopTracker,
    SuppressionController,
    SuppressionScope,
)


def test_open_loop_lifecycle():
    tracker = OpenLoopTracker()
    loop1 = tracker.register(
        title="Implement Phase III SWPC readiness check",
        description="Verify microelectronics compliance check against DoD standard",
        priority=10,
    )
    loop2 = tracker.register(
        title="Review benchmark logs",
        description="Audit FTS5 latency metrics",
        priority=20,
    )

    assert loop1.is_active
    assert loop2.is_active

    active = tracker.list_active()
    assert len(active) == 2
    assert active[0].id == loop1.id  # Higher priority (10 < 20)

    # Convert to ContextItems
    items = tracker.to_context_items()
    assert len(items) == 2
    assert all(it.dimension == CognitiveDimension.GOAL for it in items)
    assert "[Active Open Loop" in items[0].content

    # Resolve loop1
    tracker.resolve(loop1.id, resolution_summary="Completed and verified via pytest")
    assert not loop1.is_active
    retrieved = tracker.get(loop1.id)
    assert retrieved is not None
    assert retrieved.status == OpenLoopStatus.RESOLVED

    # Now only loop2 is active
    active_now = tracker.list_active()
    assert len(active_now) == 1
    assert active_now[0].id == loop2.id

    # Drop loop2
    tracker.drop(loop2.id, reason="Superseded by user directive")
    assert not loop2.is_active
    assert len(tracker.list_active()) == 0


def test_suppression_controller():
    controller = SuppressionController()

    item_clean = ContextItem(
        id="item_clean",
        role="system",
        content="Stable operational parameter: batch_size=32",
        metadata={"category": "hyperparams", "key": "batch_size"},
    )
    item_stale = ContextItem(
        id="item_stale",
        role="system",
        content="Legacy temporary experiment output for obsolete run v0.1",
        metadata={"category": "legacy_experiments", "key": "exp_v01"},
    )
    item_regex = ContextItem(
        id="item_regex",
        role="user",
        content="Please check if server 192.168.1.100 is responding",
        metadata={"category": "network"},
    )

    # 1. Category hard-mute rule
    rule_cat = controller.add_rule(
        scope=SuppressionScope.CATEGORY,
        target="legacy_experiments",
        reason="Obsolete experiment run",
        hard_mute=True,
    )

    # 2. Regex soft-damp rule
    rule_regex = controller.add_rule(
        scope=SuppressionScope.REGEX,
        target=r"192\.168\.1\.100",
        reason="Host migrated to 10.168.168.29",
        discount_factor=0.1,
        hard_mute=False,
    )

    retained, hard_muted = controller.filter_and_damp([item_clean, item_stale, item_regex])

    assert len(hard_muted) == 1
    assert hard_muted[0].id == "item_stale"

    assert len(retained) == 2
    retained_ids = [it.id for it in retained]
    assert "item_clean" in retained_ids
    assert "item_regex" in retained_ids

    # Check soft damp metadata
    regex_res = next(it for it in retained if it.id == "item_regex")
    assert regex_res.metadata["suppression_discount"] == 0.1
    assert regex_res.metadata["suppression_reason"] == "Host migrated to 10.168.168.29"

    # Revoke rule
    controller.revoke_rule(rule_cat.id)
    retained2, hard_muted2 = controller.filter_and_damp([item_stale])
    assert len(retained2) == 1
    assert len(hard_muted2) == 0


def test_suppression_ttl_expiration():
    controller = SuppressionController()
    item = ContextItem(id="temp_item", role="system", content="Brief state", metadata={"category": "temp"})

    # Rule with short TTL (0.05 seconds)
    controller.add_rule(
        scope=SuppressionScope.CATEGORY,
        target="temp",
        reason="Transient suppression",
        hard_mute=True,
        ttl_seconds=0.05,
    )

    # Immediate check: suppressed
    _, muted = controller.filter_and_damp([item])
    assert len(muted) == 1

    # Wait for expiration
    time.sleep(0.06)
    retained, muted_after = controller.filter_and_damp([item])
    assert len(retained) == 1
    assert len(muted_after) == 0


def test_behavioral_governor():
    governor = BehavioralGovernor()

    inv_cred = ContextItem(
        id="inv_cred",
        role="system",
        content="Never store secret values or credentials in conversation or deliverables.",
        dimension=CognitiveDimension.INVARIANT,
    )
    inv_mit = ContextItem(
        id="inv_mit",
        role="system",
        content="All public artifacts are released under MIT License without exception.",
        dimension=CognitiveDimension.INVARIANT,
    )

    # 1. Approved action
    dec1 = governor.evaluate(
        candidate_action="pytest tests/test_vault.py",
        active_invariants=[inv_cred, inv_mit],
    )
    assert dec1.verdict == GovernorVerdict.APPROVED
    assert dec1.evaluated_invariants == 2

    # 2. Blocked pattern action (rm -rf /)
    dec2 = governor.evaluate(
        candidate_action="rm -rf /opt/data/test",
        active_invariants=[inv_cred, inv_mit],
    )
    assert dec2.verdict == GovernorVerdict.REJECTED
    assert any("critical blocked pattern" in r for r in dec2.reasons)

    # 3. Credential leakage invariant violation
    dec3 = governor.evaluate(
        candidate_action="python3 run.py --api_key=my_secret_token",
        active_invariants=[inv_cred, inv_mit],
    )
    assert dec3.verdict == GovernorVerdict.REJECTED
    assert any("credential privacy invariant" in r for r in dec3.reasons)

    # 4. History containing revoked premise triggers review
    revoked_hist = ContextItem(
        id="hist_old_endpoint",
        role="system",
        content="Old API host: 192.168.1.1",
        metadata={"epistemic_state": "revoked"},
    )
    dec4 = governor.evaluate(
        candidate_action="curl http://internal-service/api",
        active_invariants=[inv_mit],
        candidate_history=[revoked_hist],
    )
    assert dec4.verdict == GovernorVerdict.NEEDS_REVIEW
    assert len(dec4.contradictions) > 0


def test_agent_self_history_tracker():
    tracker = AgentSelfHistoryTracker(max_records=5)

    rec1 = tracker.record_action(
        action_type="patch",
        description="Updated vault.py snapshot serializer",
        outcome=ActionOutcome.SUCCESS,
        lessons_learned="Always verify Merkle root after snapshot roundtrip",
    )
    rec2 = tracker.record_action(
        action_type="terminal",
        description="Run pytest suite",
        outcome=ActionOutcome.FAILURE,
        lessons_learned="Missing import in test_mcp_server",
    )

    recent = tracker.get_recent(limit=5)
    assert len(recent) == 2
    assert recent[0].id == rec2.id  # Newest first

    failures = tracker.get_recent(limit=5, outcome_filter=ActionOutcome.FAILURE)
    assert len(failures) == 1
    assert failures[0].id == rec2.id

    # Convert to context items
    ctx_items = tracker.to_context_items(limit=2)
    assert len(ctx_items) == 2
    assert all(it.dimension == CognitiveDimension.IDENTITY for it in ctx_items)
    assert "Lesson:" in ctx_items[0].content or "Lesson:" in ctx_items[1].content
