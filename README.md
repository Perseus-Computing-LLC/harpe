# harpe

> Deterministic context synthesis and token budgeting for AI agents.

[![Crates.io](https://img.shields.io/crates/v/harpe.svg)](https://crates.io/crates/harpe)
[![docs.rs](https://docs.rs/harpe/badge.svg)](https://docs.rs/harpe)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-informational.svg)](Cargo.toml)

Part of the **Telarch Suite** (TCI / Telarch Cognitive Infrastructure) by [Perseus Computing LLC](https://github.com/Perseus-Computing-LLC).

---

## Why Harpe exists

In multi-turn agent runs, prompts grow until they hit model limits. When that happens, naive truncation usually lops off system rules, tool schemas, or early task context, causing the agent to hallucinate or stall.

Harpe solves this by enforcing hard token budgets before calling an LLM. It sorts context items by priority and applies deterministic eviction rules so your agents never exceed token limits or drop critical instructions.

---

## Key design goals

- **Microsecond speed:** Assembling a 10-item context takes a median of 1.88 &micro;s; 200 items take 41.16 &micro;s (see [Benchmark results](#benchmark-results)).
- **Predictable eviction:** Policies like `LowestPriority`, `Fifo`, `TruncateTail`, and `LeastRelevant` ensure prompt assembly is fully reproducible.
- **Hard ceiling:** The assembled context never exceeds the available budget; if pinned items alone do not fit, `synthesize` returns an error. In 100,000 simulated turns, 0 went over.
- **No network calls:** Runs fully offline on your own machine with zero cloud dependencies.

---

## Installation

Add `harpe` to your `Cargo.toml`:

```toml
[dependencies]
harpe = "0.1.0-alpha.2"
```

Or via Cargo:

```bash
cargo add harpe
```

---

## Example

```rust
use harpe::{ContextBudgetEngine, ContextItem, ContextSynthesizer, PrunePolicy, TokenBudget};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Set budget boundaries
    let budget = TokenBudget {
        total_budget: 4096,
        reserved_completion: 1024,
        reserved_tools: 512,
        safety_margin: 128,
    };

    assert_eq!(budget.available_context(), 2432);

    // 2. Set up the engine with a token counter (or use default)
    let engine = ContextBudgetEngine::default();

    // 3. Assemble candidate items with priorities (priority 0 is pinned)
    let items = vec![
        ContextItem::new(
            "system_directive",
            "You are a verification agent. Run all tests.",
            0,
        ),
        ContextItem::new(
            "tool_output_1",
            "Execution trace: node 4 state reconciled.",
            50,
        ),
        ContextItem::new(
            "historical_logs",
            "Verbose background data stream...",
            80,
        ),
    ];

    // 4. Synthesize prompt within budget
    let report = engine.synthesize(&items, &budget, PrunePolicy::LowestPriority)?;

    println!("Consumed Tokens: {}", report.consumed_tokens);
    println!("Included Items:  {}", report.included_items.len());
    println!("Pruned Items:    {}", report.pruned_items.len());

    assert!(report.consumed_tokens <= budget.available_context());

    Ok(())
}
```

---

## Python: keeping confidential items off the wire

Mark an item `sensitivity="confidential"` and `synthesize(...)` leaves it out whenever the destination is `"remote"` (the default), even if the item is pinned. Withheld items cost no budget and are listed in `report.withheld_items`, so you can see what was held back. Pass `destination="local"` for a model that runs on your own hardware.

```python
from harpe import ContextBudgetEngine, ContextItem, TokenBudget

engine = ContextBudgetEngine()
items = [
    ContextItem(id="rule", content="Answer in plain language.", pinned=True),
    ContextItem(id="note", content="Internal-only detail.", sensitivity="confidential"),
]
report = engine.synthesize(items, TokenBudget(total_budget=1000))                      # note withheld
local = engine.synthesize(items, TokenBudget(total_budget=1000), destination="local")  # note kept
```

## Benchmark results

Measured with the [`telarch-benchmarks`](https://github.com/Perseus-Computing-LLC/telarch-benchmarks) suite on bare-metal Linux x86_64 (Intel Core i5-13500, runs pinned to the performance cores, `rustc 1.97.1`, default release profile). The figures come from that repository's committed reports.

| Metric | Measurement | Test condition | Operational Guarantee |
| :--- | :--- | :--- | :--- |
| **Small Context Assembly (p50)** | **1.88 &micro;s** | 10 items, 60-token budget, 2,000 iterations (p95 2.36 &micro;s, p99 3.16 &micro;s) | Deterministic assembly |
| **Large Context Assembly (p50)** | **41.16 &micro;s** | 200 items, 1,500-token budget, 500 iterations | Priority-ordered greedy selection under a hard ceiling |
| **100,000-Turn Production Curve** | **46.9 &micro;s / turn** | 100,000 sequential turns, 1,200-token ceiling (`knapsack_cost_curve`) | ~21,300 turns/s sustained throughput |
| **Context Token Reduction** | **78.28%** | 100,000 simulated turns against unbudgeted accumulation of the same items | Every turn stays under the ceiling |
| **Budget Overruns** | **0 (0.00%)** | 100,000 evaluated turns | Fail-closed ceiling |
| **Tier 0 Tactical Payload** | **20.33 &micro;s** | 800 tokens to 51 tokens (93.6% reduction), 60-token ceiling (`restricted_payload_test`) | 79 bytes binary / 132 bytes wire JSON |

Run benchmarks yourself:

```bash
cargo run --release -p telarch-benchmarks
cargo run --release -p telarch-benchmarks --bin knapsack_cost_curve
cargo run --release -p telarch-benchmarks --bin restricted_payload_test
```

---

## License

Copyright &copy; 2026 Perseus Computing LLC. Released under the [MIT License](LICENSE).
