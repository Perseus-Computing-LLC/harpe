# Harpe

> Deterministic context synthesis and token budgeting for cognitive AI agents.

[![Crates.io](https://img.shields.io/crates/v/harpe.svg)](https://crates.io/crates/harpe)
[![PyPI](https://img.shields.io/pypi/v/harpe.svg)](https://pypi.org/project/harpe/)
[![docs.rs](https://docs.rs/harpe/badge.svg)](https://docs.rs/harpe)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![MSRV: 1.85](https://img.shields.io/badge/MSRV-1.85-informational.svg)](Cargo.toml)

Part of the **Telarch Suite** by [Perseus Computing LLC](https://perseuscomputing.com).

## What it does

In multi-turn agent runs, prompt context grows until it exceeds model limits. Harpe selects which items go into each model call and keeps the total under a strict token ceiling:
- **Priority-ordered greedy selection:** Evaluates items by priority number while total tokens fit within the budget ceiling.
- **Pinned items always stay:** Priority 0 items (or items marked pinned) are never dropped.
- **Deterministic output:** Identical input items and ceiling yield identical token selections every time.
- **Zero network code:** Operates purely in-memory on your own hardware without external calls or telemetry.

## Installation

### Rust

Add `harpe` to `Cargo.toml`:

```toml
[dependencies]
harpe = "0.1.0-alpha.3"
```

### Python

Install via pip:

```bash
pip install harpe
```

## Quickstart

### Rust (10 lines)

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

### Python (10 lines)

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

## What we measured

Figures trace to reports committed in [telarch-benchmarks](https://github.com/Perseus-Computing-LLC/telarch-benchmarks) (Linux x86_64, Intel Core i5-13500, `rustc 1.97.1`):

| Metric | Result | Range | Scope | Method |
|---|---|---|---|---|
| **Small Context Latency (p50)** | **1.88 µs** | 1.69–2.30 µs | 10 items, 60-token ceiling | `telarch-benchmarks` |
| **Large Context Latency (p50)** | **41.16 µs** | 37.68–43.51 µs | 200 items, 1,500-token ceiling | `telarch-benchmarks` |
| **Context Reduction Ratio** | **69.51%** | 69.51% | 200 items against 1,500 ceiling | `telarch-benchmarks` |
| **100,000-Turn Production Curve** | **46.90 µs/turn** | 44.6–49.6 µs | 100,000 simulated agent turns | `knapsack_cost_curve` |
| **Turn Token Reduction** | **78.28%** | 78.28% | 100,000 turns, 1,200 ceiling | `knapsack_cost_curve` |
| **Ceiling Overruns** | **0 turns** | 0 turns | 100,000 turns, 1,200 ceiling | `knapsack_cost_curve` |
| **Fact Retention Accuracy** | **100.0%** | 100.0% | 50 tasks over 3 seeds (400 ceiling) | `gap1_harpe_quality` |
| **LLM Turn Latency Share** | **0.00038%** | - | Share of 500 ms LLM invocation | `gap2_harpe_alternatives` |

## Limits

- **Alpha release:** APIs and structures are subject to change before 1.0.
- **Not an optimal 0/1 knapsack solver:** Harpe uses priority-ordered greedy selection, not dynamic programming or integer linear programming.
- **Allocations:** Harpe allocates context items and report structures during synthesis.
- **Local token heuristic:** The default counter approximates token counts using whitespace and character length heuristics. When exact token bounds are required with downstream providers, pass a tokenizer-specific counter.
- **Not independently assessed:** Designed for local and disconnected use.

## Links

- [Documentation (docs.rs)](https://docs.rs/harpe)
- [Evidence & Claims](https://telarch.ai/evidence)
- [Security Policy](https://github.com/Perseus-Computing-LLC/.github/blob/main/SECURITY.md)
- [Telarch Overview](https://telarch.ai)

## License

Released under the [MIT License](LICENSE) © 2026 Perseus Computing LLC.
