"""H-A (issue #11): the exact 0/1 knapsack kernel behind ``PrunePolicy.KNAPSACK_DP`` runs in the Rust crate
through the ``harpe_native`` extension (bindings/python) when it is importable, and in Python otherwise.

Both kernels must choose the same set on every instance: the same strict-improvement tie rule, the same
float64 additions in the same order. Without the extension these tests skip on a developer machine and
fail in CI (CI=true), so a missing build can never pass as agreement.
"""

import json
import os
import random
import unittest
from unittest import mock

from harpe import ContextBudgetEngine, ContextItem, PrunePolicy, TokenBudget
from harpe import _native
from test_knapsack_fast_hb import make_item, random_instance, reference_knapsack

PURE = "HARPE_PURE_PYTHON"


def _skip_or_fail(case, reason):
    if os.environ.get("CI", "").lower() in ("1", "true"):
        case.fail(f"{reason} (CI must build the extension before pytest)")
    case.skipTest(reason)


class NativeKnapsackTest(unittest.TestCase):
    def setUp(self):
        if not _native.available():
            _skip_or_fail(self, "harpe_native extension not built (bindings/python, maturin develop)")
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop(PURE, None)

    # --- kernel agreement --------------------------------------------------------------------

    def test_kernel_matches_python_and_the_textbook_table_on_random_instances(self):
        rng = random.Random(20261029)
        for trial in range(60):
            n = rng.randint(1, 40)
            capacity = rng.randint(1, 120)
            weights, utilities = random_instance(rng, n, capacity)
            fit = [i for i in range(n) if 1 <= weights[i] <= capacity]
            if not fit:
                continue
            w = [weights[i] for i in fit]
            u = [utilities[i] for i in fit]
            with self.subTest(trial=trial, n=len(w), capacity=capacity):
                native = set(_native.exact_knapsack(w, u, capacity))
                self.assertEqual(native, ContextBudgetEngine._exact_knapsack(w, u, capacity))
                self.assertEqual(native, reference_knapsack(w, u, capacity))

    def test_kernel_ties_resolve_to_earlier_items(self):
        # Three equal items, room for two: the first two win, in both kernels.
        self.assertEqual(sorted(_native.exact_knapsack([5, 5, 5], [1.0, 1.0, 1.0], 10)), [0, 1])
        self.assertEqual(ContextBudgetEngine._exact_knapsack([5, 5, 5], [1.0, 1.0, 1.0], 10), {0, 1})

    def test_kernel_float_order_is_the_python_order(self):
        # Sums that differ by rounding order: the kernels must add dp[c - w] + u exactly as Python does.
        w = [1, 1, 1, 1]
        u = [0.1, 0.2, 0.3, 1e16]
        self.assertEqual(set(_native.exact_knapsack(w, u, 3)), ContextBudgetEngine._exact_knapsack(w, u, 3))

    def test_kernel_rejects_malformed_input(self):
        with self.assertRaises(ValueError):
            _native.exact_knapsack([1, 2], [1.0], 3)  # length mismatch
        with self.assertRaises(ValueError):
            _native.exact_knapsack([0, 2], [1.0, 1.0], 3)  # weight below 1
        with self.assertRaises(ValueError):
            _native.exact_knapsack([4, 2], [1.0, 1.0], 3)  # weight above capacity
        self.assertEqual(_native.exact_knapsack([], [], 5), [])

    # --- dispatch ------------------------------------------------------------------------------

    def _items(self, n=30, seed=7):
        rng = random.Random(seed)
        return [make_item(i, rng.randint(1, 400)) for i in range(n)]

    def test_knapsack_policy_uses_the_native_kernel_when_available(self):
        items = self._items()
        with mock.patch.object(_native, "exact_knapsack", wraps=_native.exact_knapsack) as spy:
            report = ContextBudgetEngine().synthesize(items, TokenBudget(total_budget=3000), policy=PrunePolicy.KNAPSACK_DP)
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(report.knapsack_solver, "native")

    def test_env_override_forces_the_python_kernel(self):
        items = self._items()
        with mock.patch.dict(os.environ, {PURE: "1"}), mock.patch.object(_native, "exact_knapsack") as spy:
            report = ContextBudgetEngine().synthesize(items, TokenBudget(total_budget=3000), policy=PrunePolicy.KNAPSACK_DP)
        spy.assert_not_called()
        self.assertEqual(report.knapsack_solver, "python")

    def test_reports_are_identical_between_kernels_except_the_solver_name(self):
        for policy in (PrunePolicy.KNAPSACK_DP, PrunePolicy.DUO_ATTENTION):
            for seed in range(5):
                with self.subTest(policy=policy.value, seed=seed):
                    items = self._items(seed=seed)
                    budget = TokenBudget(total_budget=2500)
                    native = ContextBudgetEngine().synthesize(items, budget, policy=policy, now=100.0)
                    with mock.patch.dict(os.environ, {PURE: "1"}):
                        pure = ContextBudgetEngine().synthesize(items, budget, policy=policy, now=100.0)
                    self.assertEqual((native.knapsack_solver, pure.knapsack_solver), ("native", "python"))
                    n, p = native.model_dump(), pure.model_dump()
                    n.pop("knapsack_solver"), p.pop("knapsack_solver")
                    self.assertEqual(json.dumps(n, sort_keys=True, default=str), json.dumps(p, sort_keys=True, default=str))

    def test_greedy_fallback_above_the_cell_cap_stays_in_python(self):
        items = self._items()
        with mock.patch.object(_native, "exact_knapsack") as spy:
            sel, pruned = ContextBudgetEngine._solve_knapsack_dp(items, [1.0] * len(items), 3000, max_cells=10)
        spy.assert_not_called()
        self.assertEqual(len(sel) + len(pruned), len(items))
        report = ContextBudgetEngine().synthesize(items, TokenBudget(total_budget=3000), policy=PrunePolicy.LOWEST_PRIORITY)
        self.assertIsNone(report.knapsack_solver)

    def test_native_reports_its_crate_version(self):
        self.assertRegex(_native.crate_version(), r"^\d+\.\d+\.\d+")


if __name__ == "__main__":
    unittest.main()
