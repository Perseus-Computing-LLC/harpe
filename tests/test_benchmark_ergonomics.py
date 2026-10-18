"""Unit tests for benchmark ergonomics and parameter flexibility in Harpe."""

import unittest
from harpe import (
    ContextBudgetEngine,
    ContextItem,
    PrunePolicy,
    TokenBudget,
    format_official_cot_prompt,
    format_qa_prompt,
)


class TestHarpeBenchmarkErgonomics(unittest.TestCase):
    def setUp(self):
        self.engine = ContextBudgetEngine()

    def test_context_item_item_id_alias(self):
        item = ContextItem(item_id="q-42", content="Retrieved evidence memory")
        self.assertEqual(item.id, "q-42")
        self.assertEqual(item.item_id, "q-42")
        self.assertEqual(item.content, "Retrieved evidence memory")

    def test_context_item_auto_id_fallback(self):
        item = ContextItem(content="Memory without explicit ID")
        self.assertTrue(len(item.id) > 0)
        self.assertEqual(item.item_id, item.id)

    def test_token_budget_system_reserve_aliases(self):
        budget = TokenBudget(total_budget=2500, system_reserve=200, tool_reserve=50)
        self.assertEqual(budget.safety_margin, 200)
        self.assertEqual(budget.system_reserve, 200)
        self.assertEqual(budget.reserved_tools, 50)
        self.assertEqual(budget.tool_reserve, 50)
        self.assertEqual(budget.available_context, 2250)

    def test_engine_synthesize_prune_policy_kwarg(self):
        budget = TokenBudget(total_budget=50)
        items = [
            ContextItem(id="prefix1", content="Prefix prompt", priority=50, created_at=100.0, token_count=20),
            ContextItem(id="mid1", content="Intermediate scratch", priority=50, created_at=200.0, token_count=20),
            ContextItem(id="tail1", content="Latest question", priority=50, created_at=300.0, token_count=20),
        ]
        report = self.engine.synthesize(items, budget, prune_policy=PrunePolicy.WARM_PREFIX)
        self.assertEqual(report.policy_applied, PrunePolicy.WARM_PREFIX)
        self.assertIn("prefix1", [it.id for it in report.included_items])
        self.assertIn("tail1", [it.id for it in report.included_items])

    def test_synthesis_report_render_and_to_context_str(self):
        budget = TokenBudget(total_budget=100)
        items = [
            ContextItem(id="1", content="Turn 1: User likes coffee"),
            ContextItem(id="2", content="Turn 2: User works in defense"),
        ]
        report = self.engine.synthesize(items, budget)
        rendered = report.to_context_str()
        self.assertEqual(rendered, "Turn 1: User likes coffee\n\nTurn 2: User works in defense")
        self.assertEqual(report.render(), rendered)

    def test_format_official_cot_prompt(self):
        # Test keyword arguments
        prompt = format_official_cot_prompt(
            retrieved_context="Fact A\nFact B",
            question_content="What did user say?",
            question_date="2026-10-17",
        )
        self.assertIn("<context>\nFact A\nFact B\n</context>", prompt)
        self.assertIn("<question_date> 2026-10-17 </question_date>", prompt)
        self.assertIn("<question>\n  What did user say?\n</question>", prompt)
        self.assertTrue(prompt.endswith("Answer:"))

        # Test positional arguments (context, date, content)
        prompt_pos = format_official_cot_prompt("Fact A", "2026-10-17", "What did user say?")
        self.assertIn("<context>\nFact A\n</context>", prompt_pos)
        self.assertIn("<question_date> 2026-10-17 </question_date>", prompt_pos)
        self.assertIn("<question>\n  What did user say?\n</question>", prompt_pos)

    def test_format_qa_prompt(self):
        prompt = format_qa_prompt("Context snippet", "Where is the file?")
        self.assertIn("Context information is below.", prompt)
        self.assertIn("Context snippet", prompt)
        self.assertIn("Where is the file?", prompt)


if __name__ == "__main__":
    unittest.main()
