"""The speed-ups taken from the third review of PR #16 (review_notes/R3_PR16.md, Task D) change no result.

Each test compares the code in ``src/harpe`` with the definition it had before the speed-up, written out here.
"""
from __future__ import annotations

import random
import re
from typing import List, Tuple

from harpe import ContextBudgetEngine, ContextItem
from harpe.engine import HeuristicTokenCounter

_PIECE_RE = re.compile(r"[A-Za-z0-9_]+|\S")
_HEX_RUN_RE = re.compile(r"[0-9a-fA-F]{8,}")


def _count_in_two_passes(text: str) -> int:
    """``HeuristicTokenCounter.count_tokens`` as it was: pieces over the text, then hex runs over the text."""
    if not text:
        return 0
    pieces = len(_PIECE_RE.findall(text))
    pieces += sum(len(run) // 3 for run in _HEX_RUN_RE.findall(text))
    return max(pieces, (len(text) + 3) // 4)


def _texts() -> List[str]:
    rng = random.Random(20261110)
    parts = ["a", "f", "F", "0", "9", "g", "G", "_", " ", "\n", "\t", "-", ":", "\u00e9", "\u6f22", "\U0001F600",
             "\u0661", "\uff11", "\uff41", "deadbeef", "DEADBEEF", "0123456789abcdef", "x" * 9, "cafe", "abcdef0", "\u2028"]
    texts = ["".join(rng.choice(parts) for _ in range(rng.randint(0, 40))) for _ in range(20000)]
    texts += [
        "", " ", "deadbeef", "deadbee", "deadbeefg", "gdeadbeef", "dead_beef", "deadbeef_deadbeef", "deadbeef-deadbeef",
        "0123456789abcdef0123456789abcdef", "sha256:" + "ab12" * 16, "x" * 7 + "deadbeef", "deadbeef" + "x" * 7,
        "commit 9fceb02d0ae598e95dc970b74767f19372d61af8 touched 12 files", "\u00e9deadbeef\u00e9", "deadbeef\u6f22cafebabe",
        "\uff41\uff42\uff43\uff44\uff45\uff46\uff10\uff11",  # full-width letters and digits: not ASCII, so not a hex run
        "word " * 500, "0123456789abcdef " * 200, ("f" * 8 + "g") * 100,
    ]
    return texts


def test_count_tokens_equals_the_two_pass_definition():
    count = HeuristicTokenCounter().count_tokens
    for text in _texts():
        assert count(text) == _count_in_two_passes(text), repr(text)


def _greedy_with_tuple_keys(weights: List[int], utilities: List[float], capacity: int) -> List[int]:
    """The greedy fallback of ``_solve_knapsack_dp`` as it was: both orders sorted on ``(key, index)`` tuples."""
    n = len(weights)
    take, used = set(), 0
    for i in sorted(range(n), key=lambda i: (-utilities[i] / weights[i], i)):
        if used + weights[i] <= capacity:
            take.add(i)
            used += weights[i]
    for i in sorted(range(n), key=lambda i: (-utilities[i], i)):
        if i not in take and used + weights[i] <= capacity:
            take.add(i)
            used += weights[i]
    return sorted(take)


def _greedy_cases() -> List[Tuple[List[int], List[float], int]]:
    rng = random.Random(20261110)
    nan, inf = float("nan"), float("inf")
    cases: List[Tuple[List[int], List[float], int]] = []
    for round_ in range(600):
        n = rng.randint(1, 60)
        weights = [rng.choice([1, 2, 3, 5, 8, 13, 40, 100]) for _ in range(n)]
        kind = round_ % 6
        if kind == 0:  # many equal densities and utilities: the tie-break decides
            utilities = [float(w) * rng.choice([1.0, 2.0]) for w in weights]
        elif kind == 1:
            utilities = [rng.choice([0.0, -0.0, 1.0, 1.0, 2.5]) for _ in range(n)]
        elif kind == 2:
            utilities = [rng.uniform(-5.0, 50.0) for _ in range(n)]
        elif kind == 3:
            utilities = [rng.choice([1.0, 7.0, inf, -inf]) for _ in range(n)]
        elif kind == 4:  # a NaN compares as neither smaller nor larger
            utilities = [rng.choice([1.0, 3.0, 3.0, nan, 9.0]) for _ in range(n)]
        else:
            utilities = [float(rng.randint(0, 3)) for _ in range(n)]
        cases.append((weights, utilities, rng.randint(1, max(2, sum(weights)))))
    for n in (700, 1500, 3000):  # long enough for the sort to merge runs, with ties throughout
        weights = [rng.choice([1, 2, 3, 5, 8]) for _ in range(n)]
        cases.append((weights, [float(w) * rng.choice([1.0, 2.0, 2.0]) for w in weights], sum(weights) // 3))
        cases.append((weights, [rng.choice([0.0, 1.0, 1.0, 4.0, nan]) for _ in range(n)], sum(weights) // 2))
    return cases


def test_the_greedy_fallback_selects_what_the_tuple_keys_selected():
    for weights, utilities, capacity in _greedy_cases():
        items = [ContextItem(id=f"i{k}", content="x", token_count=w) for k, w in enumerate(weights)]
        trace: dict = {}
        chosen, rejected = ContextBudgetEngine._solve_knapsack_dp(items, utilities, capacity, max_cells=0, trace=trace)
        if not any(w <= capacity for w in weights):
            assert chosen == []  # nothing fits: decided before either solver
            continue
        assert trace == {"solver": "greedy"}
        expected = _greedy_with_tuple_keys(weights, utilities, capacity)
        assert [int(it.id[1:]) for it in chosen] == expected, (weights, utilities, capacity)
        assert [it.id for it in chosen + rejected].count("i0") == 1 and len(chosen) + len(rejected) == len(items)
