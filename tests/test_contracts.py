"""Unit tests for Harpe golden contract and context budget engine."""

import unittest
from harpe import (
    ContextBudgetEngine,
    ContextItem,
    HeuristicTokenCounter,
    PrunePolicy,
    SynthesisReport,
    TokenBudget,
)


class TestHarpeContracts(unittest.TestCase):
    def setUp(self):
        self.counter = HeuristicTokenCounter()
        self.engine = ContextBudgetEngine(token_counter=self.counter)

    def test_token_budget_calculation(self):
        budget = TokenBudget(
            total_budget=4000,
            reserved_completion=1000,
            reserved_tools=500,
            safety_margin=100,
        )
        self.assertEqual(budget.available_context, 2400)
        self.assertFalse(budget.is_exceeded(2400))
        self.assertTrue(budget.is_exceeded(2401))
        self.assertAlmostEqual(budget.utilization_pct(1200), 50.0)

    def test_token_counter_heuristic(self):
        count_empty = self.counter.count_tokens("")
        self.assertEqual(count_empty, 0)
        count_text = self.counter.count_tokens("Hello world! This is a test.")
        self.assertGreater(count_text, 0)

    def test_synthesis_priority_pruning(self):
        budget = TokenBudget(total_budget=100)
        items = [
            ContextItem(id="item1", content="Essential system rule", priority=0, pinned=True, token_count=20),
            ContextItem(id="item2", content="High priority fact", priority=10, token_count=30),
            ContextItem(id="item3", content="Medium priority observation", priority=40, token_count=30),
            ContextItem(id="item4", content="Low priority noise", priority=90, token_count=40),
        ]
        report = self.engine.synthesize(items, budget, policy=PrunePolicy.LOWEST_PRIORITY)
        included_ids = [it.id for it in report.included_items]
        pruned_ids = [it.id for it in report.pruned_items]

        self.assertIn("item1", included_ids)
        self.assertIn("item2", included_ids)
        self.assertIn("item3", included_ids)
        self.assertIn("item4", pruned_ids)
        self.assertEqual(report.consumed_tokens, 80)
        self.assertEqual(report.budget_remaining, 20)
        self.assertEqual(report.pruned_tokens, 40)

    def test_synthesis_fifo_pruning(self):
        budget = TokenBudget(total_budget=50)
        items = [
            ContextItem(id="old", content="Old turn", priority=50, created_at=100.0, token_count=25),
            ContextItem(id="mid", content="Mid turn", priority=50, created_at=200.0, token_count=25),
            ContextItem(id="new", content="New turn", priority=50, created_at=300.0, token_count=25),
        ]
        report = self.engine.synthesize(items, budget, policy=PrunePolicy.FIFO)
        included_ids = [it.id for it in report.included_items]
        pruned_ids = [it.id for it in report.pruned_items]

        # Newer turns kept over older turns
        self.assertIn("new", included_ids)
        self.assertIn("mid", included_ids)
        self.assertIn("old", pruned_ids)


if __name__ == "__main__":
    unittest.main()
