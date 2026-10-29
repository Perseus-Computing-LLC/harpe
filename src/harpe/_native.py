"""Optional bridge to the ``harpe_native`` extension (bindings/python, pyo3).

When the extension is importable, :meth:`ContextBudgetEngine._solve_knapsack_dp` solves the exact
0/1 DP in the Rust kernel (``harpe::exact_knapsack``) and the report carries
``knapsack_solver="native"``; otherwise, or with ``HARPE_PURE_PYTHON=1`` set, the Python kernel
runs and the report says ``"python"``. Both kernels return the same selection on every instance:
the same strict-improvement tie rule and the same float64 additions in the same order.
``tests/test_native_knapsack.py`` holds them to that."""

import os
from typing import List, Sequence

PURE_PYTHON_ENV = "HARPE_PURE_PYTHON"

try:  # pragma: no cover - exercised only where the extension is built
    import harpe_native as _ext
except ImportError:  # pragma: no cover
    _ext = None


def available() -> bool:
    """True if the extension module is importable in this interpreter."""
    return _ext is not None


def enabled() -> bool:
    """True if the exact DP should run in the extension: importable and not opted out."""
    return available() and os.environ.get(PURE_PYTHON_ENV, "").strip().lower() not in ("1", "true", "yes")


def exact_knapsack(weights: Sequence[int], utilities: Sequence[float], capacity: int) -> List[int]:
    """Indices (ascending) chosen by the Rust kernel; every weight must lie in ``1..=capacity``."""
    if _ext is None:
        raise RuntimeError("harpe_native is not installed")
    return _ext.exact_knapsack(list(weights), [float(u) for u in utilities], capacity)


def crate_version() -> str:
    if _ext is None:
        raise RuntimeError("harpe_native is not installed")
    return _ext.crate_version()
