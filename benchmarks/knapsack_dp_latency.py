#!/usr/bin/env python3
"""Latency of the exact 0/1 knapsack kernel behind ``PrunePolicy.KNAPSACK_DP`` (issue #11).

Times ``ContextBudgetEngine._exact_knapsack`` (Python) and ``harpe_native.exact_knapsack`` (the Rust
crate through pyo3) on the same seeded instances, at sizes from a small plugin turn up to the
5,000,000-cell cap above which the engine falls back to greedy. Reports best-of-N wall time per
kernel and checks that the two kernels chose the same set on every instance.

Usage: python benchmarks/knapsack_dp_latency.py [--repeat N] [--out FILE.json]
"""

import argparse
import json
import platform
import random
import statistics
import sys
import time

from harpe import ContextBudgetEngine, __version__
from harpe import _native

# (label, n items, capacity): n * (capacity + 1) cells. Token counts are drawn as a plugin turn would
# see them: mostly 20..400 tokens with a few long items; capacities are the unpinned headroom.
SIZES = [
    ("plugin-turn   n=40   cap=4,000  (160k cells)", 40, 4_000),
    ("large-turn    n=120  cap=8,000  (960k cells)", 120, 8_000),
    ("wide-window   n=200  cap=16,000 (3.2M cells)", 200, 16_000),
    ("at-the-cap    n=250  cap=19,999 (5.0M cells)", 250, 19_999),
]


def instance(rng, n, capacity):
    weights = []
    for _ in range(n):
        roll = rng.random()
        if roll < 0.1:
            weights.append(rng.randint(400, min(capacity, 2_000)))
        else:
            weights.append(rng.randint(20, 400))
    utilities = [round(rng.random() * 100.0, 3) for _ in range(n)]
    return weights, utilities


def best_of(fn, repeat):
    times = []
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return min(times), statistics.median(times), result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=5, choices=range(1, 101), metavar="N")
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=20261029)
    args = ap.parse_args()

    rows = []
    for label, n, capacity in SIZES:
        rng = random.Random(args.seed)
        weights, utilities = instance(rng, n, capacity)
        assert n * (capacity + 1) <= 5_000_000
        py_best, py_med, py_set = best_of(lambda: ContextBudgetEngine._exact_knapsack(weights, utilities, capacity), args.repeat)
        row = {
            "label": label, "n": n, "capacity": capacity, "cells": n * (capacity + 1),
            "python_best_s": round(py_best, 6), "python_median_s": round(py_med, 6),
        }
        if _native.available():
            na_best, na_med, na_list = best_of(lambda: _native.exact_knapsack(weights, utilities, capacity), args.repeat)
            if set(na_list or []) != py_set:
                print(f"KERNEL DISAGREEMENT on {label}", file=sys.stderr)
                sys.exit(2)
            row.update({
                "native_best_s": round(na_best, 6), "native_median_s": round(na_med, 6),
                "speedup_best": round(py_best / na_best, 1), "same_selection": True,
            })
        rows.append(row)
        print(f"{label}  python {py_best*1000:8.1f} ms" + (f"  native {row['native_best_s']*1000:7.2f} ms  x{row['speedup_best']}" if "native_best_s" in row else ""))

    out = {
        "harpe": __version__,
        "native": _native.crate_version() if _native.available() else None,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "host": "developer container (not the EC2 benchmark harness); see telarch-benchmarks for published engine figures",
        "repeat": args.repeat,
        "seed": args.seed,
        "note": "best-of-N wall time per kernel on identical seeded instances; both kernels chose the same set on every row",
        "rows": rows,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
            fh.write("\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
