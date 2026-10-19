# Harpe

> Context packing under a token budget for AI agents.

[![Crates.io](https://img.shields.io/crates/v/harpe.svg)](https://crates.io/crates/harpe)
[![PyPI](https://img.shields.io/pypi/v/harpe.svg)](https://pypi.org/project/harpe/)
[![docs.rs](https://docs.rs/harpe/badge.svg)](https://docs.rs/harpe)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![MSRV: 1.85](https://img.shields.io/badge/MSRV-1.85-informational.svg)](Cargo.toml)

Part of the **Telarch Suite** by [Perseus Computing LLC](https://perseuscomputing.com).

## What it does

In multi-turn agent runs, prompt context grows until it no longer fits. Harpe chooses which items go into each model call so the total stays inside a token budget:

- **Pinned items stay.** Items that are pinned or at priority 0 (and, in Python, items of the invariant dimension) are always included. If they alone exceed the budget, nothing else is added and the report shows the overrun (`synthesize` in `src/harpe/engine.py` and `src/lib.rs`).
- **Greedy by priority by default.** `PrunePolicy.LOWEST_PRIORITY` sorts the other items by priority (lower number first; in Python, newer first among equals) and adds each one that still fits. Other policies order items by age, input order or a relevance score; the Python engine also has a dynamic-programming selection over a per-item utility score (`KNAPSACK_DP`).
- **Confidential items withheld.** With the default `destination="remote"`, the Python engine drops every item marked confidential before budgeting, so it never takes a slot or reaches the output (`src/harpe/engine.py`).
- **Deterministic by default.** With the default policy, the same items and budget give the same selection.
- **No network code.** Selection runs in memory and makes no network calls.

## Installation

### Rust

```toml
[dependencies]
harpe = "0.1.0-alpha.4"
```

### Python

```bash
pip install harpe
```

## Quickstart

### Rust

```rust
use harpe::{ContextBudgetEngine, ContextItem, ContextSynthesizer, PrunePolicy, TokenBudget};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let budget = TokenBudget::new(2048);
    let items = [
        ContextItem::new("sys", "You are an autonomous engineering agent.", 0),
        ContextItem::new("obs", "Memory trace step 42: state normal.", 10),
    ];
    let report = ContextBudgetEngine::default().synthesize(&items, &budget, PrunePolicy::LowestPriority)?;
    println!("Tokens: {} / {} (pruned: {})", report.consumed_tokens, budget.available_context(), report.pruned_items.len());
    Ok(())
}
```

### Python

```python
from harpe import ContextBudgetEngine, ContextItem, TokenBudget, PrunePolicy

budget = TokenBudget(total_budget=2048)
items = [
    ContextItem(id="sys", content="You are an autonomous engineering agent.", priority=0, pinned=True),
    ContextItem(id="obs", content="Memory trace step 42: state normal.", priority=10),
]
report = ContextBudgetEngine().synthesize(items, budget, PrunePolicy.LOWEST_PRIORITY)
print(f"Tokens: {report.consumed_tokens} / {budget.available_context} (pruned: {len(report.pruned_items)})")
```

## Measurements

Benchmark reports, with hardware and method, are kept in [telarch-benchmarks](https://github.com/Perseus-Computing-LLC/telarch-benchmarks) rather than repeated here.

## Limits

- **Alpha release:** APIs and structures are subject to change before 1.0.
- **Greedy is not best-fit:** The default policy can leave room that a different combination of items would fill.
- **Pinned overrun:** Pinned items are kept even when they exceed the budget.
- **Time-dependent policies:** `KNAPSACK_DP` and `DUO_ATTENTION` score items by age against the current time, so their selection can change as time passes.
- **Allocations:** Harpe allocates context items and report structures during synthesis.
- **Local token heuristic:** The default counter estimates token counts from whitespace and character length. When exact counts matter for a downstream provider, pass your own `TokenCounter`.
- **Not independently assessed:** Designed for local and disconnected use.

## Links

- [Documentation (docs.rs)](https://docs.rs/harpe)
- [Security Policy](https://github.com/Perseus-Computing-LLC/.github/blob/main/SECURITY.md)
- [Telarch Overview](https://telarch.ai)

## License

Released under the [MIT License](LICENSE) © 2026 Perseus Computing LLC.
