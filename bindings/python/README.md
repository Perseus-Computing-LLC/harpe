# harpe-native

Python extension module built from the `harpe` Rust crate with pyo3. It exposes the exact 0/1
knapsack kernel (`harpe::exact_knapsack`) that the Python engine's `PrunePolicy.KNAPSACK_DP` and
`DUO_ATTENTION` policies solve, so the Hermes plugin runs the Rust kernel with the Python kernel
kept as the fallback.

The pure-Python package does not require it. When `harpe_native` is importable,
`ContextBudgetEngine._solve_knapsack_dp` calls it for every instance under the exact-DP cell cap
and the report carries `knapsack_solver="native"`; otherwise, or with `HARPE_PURE_PYTHON=1` set,
the Python kernel runs and the report says `"python"`. The greedy fallback above the cell cap is
Python on both paths. `tests/test_native_knapsack.py` holds the two kernels to identical
selections (same strict-improvement tie rule, same float64 additions in the same order).

Build into the current environment: `pip install maturin && maturin develop --release` from
this directory (`bindings/python`). Build a wheel: `maturin build --release`.
