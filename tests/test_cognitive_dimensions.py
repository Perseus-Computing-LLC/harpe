"""Unit tests for Decomposed Cognitive Dimensions in Harpe context synthesis."""

import unittest
from harpe import (
    CognitiveDimension,
    ContextBudgetEngine,
    ContextItem,
    PrunePolicy,
    TokenBudget,
)


class TestCognitiveDimensions(unittest.TestCase):
    def setUp(self):
        self.engine = ContextBudgetEngine()

    def test_cognitive_dimension_enum_values(self):
        self.assertEqual(CognitiveDimension.INVARIANT.value, "invariant")
        self.assertEqual(CognitiveDimension.GOAL.value, "goal")
        self.assertEqual(CognitiveDimension.KNOWLEDGE.value, "knowledge")
        self.assertEqual(CognitiveDimension.IDENTITY.value, "identity")
        self.assertEqual(CognitiveDimension.CONTEXT.value, "context")

    def test_invariant_dimension_guaranteed_retention(self):
        """Operational invariant items must never be crowded out by unpinned items."""
        budget = TokenBudget(total_budget=100)
        items = [
            ContextItem(
                id="inv-1",
                content="RULE: Zero network egress permitted without cryptographic lease.",
                dimension=CognitiveDimension.INVARIANT,
                priority=80,  # Even with high priority number (disposable rank), invariant dimension protects it
                token_count=30,
            ),
            ContextItem(
                id="task-1",
                content="Task momentum item that would otherwise take budget.",
                dimension=CognitiveDimension.GOAL,
                priority=10,
                token_count=50,
            ),
            ContextItem(
                id="chat-1",
                content="Chitchat dialog history filling the rest of the window.",
                dimension=CognitiveDimension.CONTEXT,
                priority=20,
                token_count=40,
            ),
        ]
        # Total tokens = 120, budget = 100.
        report = self.engine.synthesize(items, budget, policy=PrunePolicy.LOWEST_PRIORITY)
        included_ids = [it.id for it in report.included_items]

        # Invariant must be included
        self.assertIn("inv-1", included_ids)
        # Task goal included
        self.assertIn("task-1", included_ids)
        # Chat-1 must be pruned because capacity is exhausted
        self.assertIn("chat-1", [it.id for it in report.pruned_items])

        # Dimension breakdown accounting verified
        self.assertEqual(report.dimension_counts.get("invariant"), 1)
        self.assertEqual(report.dimension_counts.get("goal"), 1)
        self.assertEqual(report.dimension_tokens.get("invariant"), 30)
        self.assertEqual(report.dimension_tokens.get("goal"), 50)

    def test_goal_dimension_knapsack_utility_boost(self):
        """Goal dimension receives momentum boost during knapsack dynamic programming."""
        budget = TokenBudget(total_budget=80)
        items = [
            ContextItem(
                id="fact-1",
                content="Fact item without active goal momentum.",
                dimension=CognitiveDimension.KNOWLEDGE,
                token_count=40,
                priority=50,
                metadata={"rrf_score": 0.5, "attention_density": 0.5},
            ),
            ContextItem(
                id="goal-1",
                content="Active user task objective.",
                dimension=CognitiveDimension.GOAL,
                token_count=40,
                priority=50,
                metadata={"rrf_score": 0.5, "attention_density": 0.5},
            ),
            ContextItem(
                id="fact-2",
                content="Another passive fact item.",
                dimension=CognitiveDimension.KNOWLEDGE,
                token_count=40,
                priority=50,
                metadata={"rrf_score": 0.5, "attention_density": 0.5},
            ),
        ]
        # Budget = 80, items are 40 each. Exactly 2 can fit.
        # Goal-1 has identical baseline metadata to fact-1 and fact-2, but gets 1.25x goal boost.
        report = self.engine.synthesize(items, budget, policy=PrunePolicy.KNAPSACK_DP)
        included_ids = [it.id for it in report.included_items]
        self.assertIn("goal-1", included_ids)
        self.assertEqual(len(report.included_items), 2)
        self.assertEqual(len(report.pruned_items), 1)

    def test_dimension_string_and_enum_compatibility(self):
        item_str = ContextItem(id="str-dim", content="Invariant policy", dimension="invariant")
        item_enum = ContextItem(id="enum-dim", content="Invariant policy", dimension=CognitiveDimension.INVARIANT)
        self.assertEqual(item_str.dimension, "invariant")
        self.assertEqual(item_enum.dimension, CognitiveDimension.INVARIANT)


if __name__ == "__main__":
    unittest.main()
