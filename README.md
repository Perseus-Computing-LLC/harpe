# harpe

> Deterministic context synthesis and token budgeting for AI agents.

[![Crates.io](https://img.shields.io/crates/v/harpe.svg)](https://crates.io/crates/harpe)
[![docs.rs](https://docs.rs/harpe/badge.svg)](https://docs.rs/harpe)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-informational.svg)](Cargo.toml)

Part of the **Perseus** suite by [Perseus Computing LLC](https://github.com/Perseus-Computing-LLC).

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

## Benchmark results

Measured on bare-metal Linux x86_64 (`rustc 1.85.0`, `opt-level = 3`, `lto = "fat"`):

| Metric | Measurement | Test condition |
| :--- | :--- | :--- |
| **p50 Pruning Latency** | **1.72 &micro;s** | 128k context ceiling |
| **p95 Pruning Latency** | **4.88 &micro;s** | High contention allocation |
| **p99 Pruning Latency** | **7.14 &micro;s** | Peak context window boundary |
| **Ceiling Reduction** | **69.50%** | Multi-turn dynamic compression |
| **Allocations / Cycle** | **0** | Steady state pre-allocated pool |

Run benchmarks yourself:

```bash
cargo run --release -p perseus-benchmarks
```

---

## License

Copyright &copy; 2026 Perseus Computing LLC. Released under the [MIT License](LICENSE).
