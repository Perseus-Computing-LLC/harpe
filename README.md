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

- **Microsecond speed:** Median pruning latency of 1.72 &micro;s across 128k context allocations adds zero noticeable overhead to agent turns.
- **Predictable eviction:** Policies like `LowestPriority`, `Fifo`, `TruncateTail`, and `LeastRelevant` ensure prompt assembly is fully reproducible.
- **Zero allocation steady state:** Token accounting avoids heap churn during rapid tool-calling loops.
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
use harpe::{ContextBudgetEngine, ContextItem, PrunePolicy, TokenBudget};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Set budget boundaries
    let budget = TokenBudget {
        total_budget: 4096,
        reserved_completion: 1024,
        reserved_tools: 512,
        safety_margin: 128,
    };

    assert_eq!(budget.available_context(), 2432);

    // 2. Set up the engine to drop lower-priority items first
    let engine = ContextBudgetEngine::new(budget, PrunePolicy::LowestPriority);

    // 3. Assemble candidate items with priorities
    let items = vec![
        ContextItem {
            id: "system_directive".to_string(),
            content: "You are a verification agent. Run all tests.".to_string(),
            token_count: 80,
            priority: 100,
            compressible: false,
        },
        ContextItem {
            id: "tool_output_1".to_string(),
            content: "Execution trace: node 4 state reconciled.".to_string(),
            token_count: 500,
            priority: 50,
            compressible: true,
        },
        ContextItem {
            id: "historical_logs".to_string(),
            content: "Verbose background data stream...".to_string(),
            token_count: 2200,
            priority: 10,
            compressible: true,
        },
    ];

    // 4. Synthesize prompt within budget
    let report = engine.synthesize(items)?;

    println!("Allocated Tokens: {}", report.allocated_tokens);
    println!("Retained Items:   {}", report.retained_items.len());
    println!("Pruned Items:     {}", report.pruned_item_ids.len());

    assert!(report.allocated_tokens <= budget.available_context());

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

Measured on bare-metal Linux x86_64 (`rustc 1.85.0`, `opt-level = 3`, `lto = "fat"`):

| Metric | Measurement | Test condition | Operational Guarantee |
| :--- | :--- | :--- | :--- |
| **Small Context Assembly (p50)** | **1.76 &micro;s** | 10 items / 2k context ceiling | Sub-2 µs deterministic assembly |
| **Large Context Assembly (p50)** | **35.74 &micro;s** | 200 items / 40k context ceiling | Hard ceiling priority knapsack pruning |
| **100,000-Turn Production Curve** | **24.3 &micro;s / turn** | 100k sequential turns (`knapsack_cost_curve`) | 41,234 turns/s sustained throughput |
| **Context Token Reduction** | **84.77%** | 504.5M unbudgeted vs 76.8M budgeted tokens | Net savings of $1,283.02 / 100k turns (Sonnet 3.5) |
| **Budget Overruns** | **0 (0.00%)** | 100,000 evaluated turns | Zero overrun fail-closed ceiling guarantee |
| **Tier 0 Tactical Payload** | **21.25 &micro;s** | 800 tokens -> 51 tokens (93.6% reduction) | 79 bytes binary / 132 bytes wire JSON |

Run benchmarks yourself:

```bash
cargo run --release -p telarch-benchmarks
cargo run --release -p telarch-benchmarks --bin knapsack_cost_curve
cargo run --release -p telarch-benchmarks --bin restricted_payload_test
```

---

## License

Copyright &copy; 2026 Perseus Computing LLC. Released under the [MIT License](LICENSE).
