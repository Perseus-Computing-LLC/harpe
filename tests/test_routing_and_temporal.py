"""Tests for Harpe query intent routing, dynamic knapsack budgeting, and temporal anchors."""

import pytest
from harpe import (
    ContextBudgetEngine,
    ContextItem,
    PrunePolicy,
    QueryIntent,
    TokenBudget,
    classify_query_intent,
    create_persona_context_item,
    has_temporal_anchor,
    is_aggregation_query,
    is_preference_query,
    is_temporal_query,
    normalize_temporal_chunk,
    resolve_dynamic_budget,
)


def test_query_intent_classification():
    # Multi-session aggregation
    assert classify_query_intent("How many times did I go to the gym?") == QueryIntent.MULTI_SESSION_AGGREGATION
    assert classify_query_intent("Calculate the total amount spent across all sessions") == QueryIntent.MULTI_SESSION_AGGREGATION
    assert is_aggregation_query("List all books mentioned")

    # Temporal reasoning
    assert classify_query_intent("What did I do before my trip to Chicago?") == QueryIntent.TEMPORAL
    assert classify_query_intent("When did I start my new job?") == QueryIntent.TEMPORAL
    assert is_temporal_query("What happened earlier that week?")

    # Preference / Persona
    assert classify_query_intent("What is my favorite restaurant?") == QueryIntent.PREFERENCE
    assert classify_query_intent("Do I prefer tea or coffee in the morning?") == QueryIntent.PREFERENCE
    assert is_preference_query("What kind of movies do I like?")

    # Factoid single
    assert classify_query_intent("What is Alice's email address?") == QueryIntent.FACTOID_SINGLE
    assert classify_query_intent("") == QueryIntent.GENERAL


def test_dynamic_budget_resolution():
    base = TokenBudget(total_budget=600, system_reserve=100, tool_reserve=0, completion_reserve=50)

    # Aggregation query should expand total_budget by factor (default 2.0x -> 1200)
    expanded = resolve_dynamic_budget(base, QueryIntent.MULTI_SESSION_AGGREGATION, aggregation_factor=2.0)
    assert expanded.total_budget == 1200
    assert expanded.system_reserve == 100
    assert expanded.completion_reserve == 50
    assert expanded.available_context == 1050

    # Non-aggregation query leaves budget unchanged
    unchanged = resolve_dynamic_budget(base, QueryIntent.FACTOID_SINGLE)
    assert unchanged.total_budget == 600


def test_temporal_normalization_and_anchor_detection():
    # Plain text without date gets ISO prefix prepended
    chunk = "Alice visited the clinic."
    norm = normalize_temporal_chunk(chunk, timestamp=1776000000)
    assert norm.startswith("[2026-")
    assert "Alice visited the clinic." in norm
    assert has_temporal_anchor(norm)

    # Text that already has bracketed date header is preserved
    already = "[2026-05-01 10:00] Meeting with Bob."
    assert normalize_temporal_chunk(already, timestamp=1776000000) == already
    assert has_temporal_anchor(already)

    # Metadata detection
    assert has_temporal_anchor("No bracketed text", metadata={"temporal_anchor": "2026-05-01"})
    assert not has_temporal_anchor("No bracketed text", metadata={"category": "general"})


def test_persona_context_item_construction():
    persona = create_persona_context_item(
        content="User prefers succinct answers and vegan recipes.",
        metadata={"user_id": "u123"},
    )
    assert persona.pinned is True
    assert persona.priority == 0
    assert persona.role == "system"
    assert persona.metadata.get("is_persona_slot") is True
    assert persona.metadata.get("user_id") == "u123"


def test_engine_synthesize_with_dynamic_budget():
    engine = ContextBudgetEngine()

    items = [
        ContextItem(id=f"item_{i}", content=f"Session entry {i} containing facts", token_count=100, priority=i+1)
        for i in range(10)
    ]
    # Base budget available context: 500 - 100 = 400 (room for 4 items)
    base_budget = TokenBudget(total_budget=500, system_reserve=100, completion_reserve=0)

    # Standard run: 4 items included
    rep_std = engine.synthesize(items, base_budget, query="What is my name?")
    assert len(rep_std.included_items) == 4
    assert rep_std.dynamic_budget_expanded is False
    assert rep_std.query_intent == QueryIntent.FACTOID_SINGLE.value

    # Aggregation query with dynamic_budget=True: expands budget, includes more items
    rep_dyn = engine.synthesize(
        items,
        base_budget,
        query="Count how many times I visited each session",
        dynamic_budget=True,
        aggregation_factor=2.0,
    )
    assert rep_dyn.dynamic_budget_expanded is True
    assert rep_dyn.query_intent == QueryIntent.MULTI_SESSION_AGGREGATION.value
    # Expanded total budget is 1000, capacity 900 -> fits 9 items
    assert len(rep_dyn.included_items) == 9


def test_engine_synthesize_temporal_boost():
    engine = ContextBudgetEngine()

    # Two items of equal baseline priority (priority=10)
    # item 1 has temporal anchor, item 2 does not
    item_temporal = ContextItem(
        id="t1",
        content="[2026-01-10 09:00] Discussed Q1 roadmap",
        token_count=100,
        priority=10,
        metadata={"temporal_anchor": "2026-01-10 09:00", "relevance": 0.5},
    )
    item_non_temporal = ContextItem(
        id="nt1",
        content="General discussion notes without timestamp",
        token_count=100,
        priority=10,
        metadata={"relevance": 0.5},
    )

    budget = TokenBudget(total_budget=250, system_reserve=100, completion_reserve=0)  # available: 150 -> fits only 1 item

    # Under temporal query, the temporal item should be prioritized over non-temporal item
    report = engine.synthesize(
        [item_non_temporal, item_temporal],
        budget,
        policy=PrunePolicy.LEAST_RELEVANT,
        query="When did we discuss the roadmap?",
        temporal_boost=1.2,
    )
    assert len(report.included_items) == 1
    assert report.included_items[0].id == "t1"
    assert report.query_intent == QueryIntent.TEMPORAL.value


def test_engine_synthesize_persona_injection():
    engine = ContextBudgetEngine()

    persona = create_persona_context_item("User dislikes dairy and prefers black coffee.")
    regular_item = ContextItem(id="r1", content="Recent orders list", token_count=50, priority=2)

    budget = TokenBudget(total_budget=300, system_reserve=50, completion_reserve=0)

    # When preference query occurs, persona is automatically injected and pinned
    report = engine.synthesize(
        [regular_item],
        budget,
        query="What kind of coffee do I usually prefer?",
        persona_item=persona,
    )
    assert report.query_intent == QueryIntent.PREFERENCE.value
    assert len(report.included_items) == 2
    assert any(it.metadata.get("is_persona_slot") is True for it in report.included_items)
