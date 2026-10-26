"""Unit tests and benchmark verification for Harpe Python Fast-Path (lines 230-235)."""

import time
import unittest
from harpe import ContextBudgetEngine, ContextItem, HeuristicTokenCounter, TokenBudget
from harpe.protocols import PruningHook


class DummyPruningHook(PruningHook):
    def pre_prune(self, items):
        pass


class TestHarpeFastPath(unittest.TestCase):
    def setUp(self):
        self.counter = HeuristicTokenCounter()
        self.engine = ContextBudgetEngine(token_counter=self.counter)

    def test_fast_path_bypasses_model_copy_when_counted_and_no_hooks(self):
        """When items have token_count > 0 and no hooks exist, model_copy is bypassed."""
        item1 = ContextItem(id="item_1", content="Alpha content", priority=10, token_count=5)
        item2 = ContextItem(id="item_2", content="Beta content", priority=20, token_count=8)
        items = [item1, item2]

        budget = TokenBudget(total_budget=100)
        report = self.engine.synthesize(items, budget)

        # Included items must be identical object references (no model_copy)
        self.assertEqual(len(report.included_items), 2)
        included_map = {it.id: it for it in report.included_items}
        self.assertIs(included_map["item_1"], item1)
        self.assertIs(included_map["item_2"], item2)

    def test_slow_path_copies_and_counts_when_uncounted(self):
        """When items have token_count <= 0, they are counted and copied."""
        item = ContextItem(id="item_uncounted", content="Gamma content with some words", priority=10, token_count=0)
        items = [item]

        budget = TokenBudget(total_budget=100)
        report = self.engine.synthesize(items, budget)

        self.assertEqual(len(report.included_items), 1)
        included_item = report.included_items[0]
        self.assertIsNot(included_item, item)
        self.assertTrue(included_item.token_count > 0)

    def test_slow_path_copies_when_hooks_registered(self):
        """When hooks are registered, items are copied to avoid in-place mutations."""
        engine_with_hook = ContextBudgetEngine(token_counter=self.counter, hooks=[DummyPruningHook()])
        item = ContextItem(id="item_hooked", content="Delta content", priority=10, token_count=10)
        items = [item]

        budget = TokenBudget(total_budget=100)
        report = engine_with_hook.synthesize(items, budget)

        self.assertEqual(len(report.included_items), 1)
        included_item = report.included_items[0]
        self.assertIsNot(included_item, item)

    def test_fast_path_overhead_reduction(self):
        """Verify the ~3-4x reduction in Pydantic overhead on 200 items."""
        items_counted = [
            ContextItem(id=f"c_{i}", content=f"Pre-counted item content payload number {i}", priority=i % 100, token_count=10)
            for i in range(200)
        ]
        items_uncounted = [
            ContextItem(id=f"u_{i}", content=f"Pre-counted item content payload number {i}", priority=i % 100, token_count=0)
            for i in range(200)
        ]
        budget = TokenBudget(total_budget=1500)

        # Warmup
        for _ in range(20):
            self.engine.synthesize(items_counted, budget)
            self.engine.synthesize(items_uncounted, budget)

        n_iter = 200
        t0 = time.perf_counter()
        for _ in range(n_iter):
            self.engine.synthesize(items_counted, budget)
        t_fast = (time.perf_counter() - t0) / n_iter

        t0 = time.perf_counter()
        for _ in range(n_iter):
            self.engine.synthesize(items_uncounted, budget)
        t_slow = (time.perf_counter() - t0) / n_iter

        speedup = t_slow / t_fast
        # Verified reduction is ~3-4x (allowing 2.5x - 6.0x margin depending on CPU load)
        self.assertGreater(speedup, 2.5, f"Expected >2.5x speedup, got {speedup:.2f}x")


if __name__ == "__main__":
    unittest.main()
