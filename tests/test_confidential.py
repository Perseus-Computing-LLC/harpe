"""Confidential items never reach a remote destination."""

import unittest

from harpe import ContextBudgetEngine, ContextItem, PrunePolicy, TokenBudget

SECRET = "Hudson-Bay-7731 internal-only rule"


def _items():
    return [
        ContextItem(id="sys", content="System rule", priority=0, pinned=True, token_count=10),
        ContextItem(id="conf-pinned", content=SECRET, priority=0, pinned=True, token_count=10, sensitivity="confidential"),
        ContextItem(id="conf", content=SECRET + " two", priority=10, token_count=10, sensitivity="confidential"),
        ContextItem(id="a", content="Public note", priority=50, token_count=10),
    ]


class TestConfidential(unittest.TestCase):
    def setUp(self):
        self.engine = ContextBudgetEngine()
        self.budget = TokenBudget(total_budget=1000)

    def _all_text(self, report):
        return " ".join(i.content for i in report.included_items + report.pruned_items)

    def test_remote_is_the_default_and_withholds_confidential(self):
        report = self.engine.synthesize(_items(), self.budget)
        self.assertNotIn("Hudson-Bay", self._all_text(report))
        self.assertEqual({i.id for i in report.withheld_items}, {"conf-pinned", "conf"})

    def test_pinned_does_not_override_confidential(self):
        report = self.engine.synthesize(_items(), self.budget, destination="remote")
        self.assertNotIn("conf-pinned", {i.id for i in report.included_items})

    def test_withheld_items_take_no_budget(self):
        report = self.engine.synthesize(_items(), self.budget, destination="remote")
        self.assertEqual(report.consumed_tokens, 20)         # sys + a only
        self.assertEqual(report.pruned_tokens, 0)

    def test_withheld_even_when_budget_is_exhausted_by_pins(self):
        tight = TokenBudget(total_budget=10)
        report = self.engine.synthesize(_items(), tight, destination="remote")
        self.assertNotIn("Hudson-Bay", self._all_text(report))
        self.assertEqual(len(report.withheld_items), 2)

    def test_local_destination_keeps_everything(self):
        report = self.engine.synthesize(_items(), self.budget, destination="local")
        self.assertEqual({i.id for i in report.included_items}, {"sys", "conf-pinned", "conf", "a"})
        self.assertEqual(report.withheld_items, [])

    def test_all_policies_withhold(self):
        for policy in PrunePolicy:
            report = self.engine.synthesize(_items(), self.budget, policy=policy, destination="remote")
            self.assertNotIn("Hudson-Bay", self._all_text(report), policy)

    def test_unmarked_items_behave_exactly_as_before(self):
        plain = [i for i in _items() if i.sensitivity == "normal"]
        self.assertEqual(
            self.engine.synthesize(plain, self.budget, destination="remote").model_dump(exclude={"withheld_items"}),
            self.engine.synthesize(plain, self.budget, destination="local").model_dump(exclude={"withheld_items"}),
        )

    def test_unknown_destination_and_sensitivity_rejected(self):
        with self.assertRaises(ValueError):
            self.engine.synthesize(_items(), self.budget, destination="cloud")
        with self.assertRaises(Exception):
            ContextItem(id="x", content="y", sensitivity="secret")


if __name__ == "__main__":
    unittest.main()
