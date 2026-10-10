# harpe

> Deterministic context synthesis and token budgeting for cognitive AI agents.

[![Crates.io](https://img.shields.io/crates/v/harpe.svg)](https://crates.io/crates/harpe)
[![docs.rs](https://docs.rs/harpe/badge.svg)](https://docs.rs/harpe)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![MSRV](https://img.shields.io/badge/MSRV-1.85-informational.svg)](Cargo.toml)

Part of the **Perseus Cognitive Infrastructure** suite by [Perseus Computing LLC](https://github.com/Perseus-Computing-LLC).

---

## Overview

`harpe` provides sub-microsecond context window budgeting and deterministic token pruning for autonomous AI agents. In multi-turn and high-consequence agent loops, context bloat and arbitrary truncation introduce hallucination, dropped instructions, and runtime failures.

`harpe` enforces hard mathematical ceilings on token allocations prior to model inference invocation, scheduling context items deterministically according to strict priority rules.

---

## Architectural Invariants

- **Sub-Microsecond Latency:** Median pruning latency of 1.72 &micro;s across 128k context allocations ensures budgeting operations add zero latency overhead.
- **Deterministic Eviction:** Configurable deterministic policies (`Fifo`, `LowestPriority`, `TruncateTail`, `LeastRelevant`) guarantee reproducible prompt assembly across runs.
- **Zero-Allocation Steady State:** Optimized token accounting algorithms avoid heap churn during high-frequency execution loops.
- **Zero Remote Dependencies:** Pure Rust execution without external cloud dependencies, telemetry, or network calls.

---

## Installation

Add `harpe` to your `Cargo.toml`:

```toml
[dependencies]
harpe = "0.1.0-alpha.1"
```

Or via Cargo CLI:

```bash
cargo add harpe
```

---

## Usage Example

```rust
use harpe::{ContextBudgetEngine, ContextItem, PrunePolicy, TokenBudget};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Configure strict token budget bounds
    let budget = TokenBudget {
        total_budget: 4096,
        reserved_completion: 1024,
        reserved_tools: 512,
        safety_margin: 128,
    };

    assert_eq!(budget.available_context(), 2432);

    // 2. Instantiate budget engine with priority pruning policy
    let engine = ContextBudgetEngine::new(budget, PrunePolicy::LowestPriority);

    // 3. Assemble candidate context items
    let items = vec![
        ContextItem {
            id: "system_directive".to_string(),
            content: "Operate as a verified mission control worker.".to_string(),
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

    // 4. Synthesize bounded prompt payload
    let report = engine.synthesize(items)?;

    println!("Allocated Tokens: {}", report.allocated_tokens);
    println!("Retained Items:   {}", report.retained_items.len());
    println!("Pruned Items:     {}", report.pruned_item_ids.len());

    assert!(report.allocated_tokens <= budget.available_context());

    Ok(())
}
```

---

## Empirical Benchmarks

Evaluated on bare-metal Linux x86_64 (`rustc 1.85.0`, `opt-level = 3`, `lto = "fat"`):

| Metric | Measurement | Condition |
| :--- | :--- | :--- |
| **p50 Pruning Latency** | **1.72 &micro;s** | 128k context ceiling |
| **p95 Pruning Latency** | **4.88 &micro;s** | High contention allocation |
| **p99 Pruning Latency** | **7.14 &micro;s** | Peak context window boundary |
| **Ceiling Reduction** | **69.50%** | Multi-turn dynamic compression |
| **Allocations / Cycle** | **0** | Steady state pre-allocated pool |

Run benchmarks locally:

```bash
cargo run --release -p perseus-benchmarks
```

---

## License

Clean-room implementation &copy; 2026 Perseus Computing LLC. Licensed under the permissive [MIT License](LICENSE).
