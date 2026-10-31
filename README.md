# Harpe

**Choose what goes into each model call, under a token budget.** Harpe picks which context items go into each call: pinned items always, then the rest by priority or recency while they fit. It reports exactly what it kept and what it dropped, and the same input gives the same result.

[![CI](https://github.com/Perseus-Computing-LLC/harpe/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/Perseus-Computing-LLC/harpe/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![MSRV: 1.85](https://img.shields.io/badge/MSRV-1.85-informational.svg)](Cargo.toml)

Harpe is one of the three [Telarch](https://telarch.ai) libraries from [Perseus Computing LLC](https://perseuscomputing.com). [Kibisis](https://github.com/Perseus-Computing-LLC/kibisis) keeps the agent's memory, and [Argus](https://github.com/Perseus-Computing-LLC/argus) keeps a tamper-evident record of what it did. Each one works on its own.

## Why

Agents collect instructions, memory, documents and tool output faster than a model can read them. Sending everything costs money and eventually fails. Trimming by hand drops the wrong things, and nobody notices. Harpe makes the cut explicit, repeatable and visible in a report.

## What you get

- **Pinned items stay.** Pinned and priority-0 items are always included. If they alone exceed the budget, nothing else is added and the report shows the overrun.
- **Room reserved first.** The Rust `TokenBudget` sets aside tokens for the answer, for tool calls and for a safety margin before anything is selected.
- **A report.** You get the kept items, the dropped items, the tokens used and the tokens left.
- **The same answer every time.** With the default policy there is no randomness and no clock, so the same items and budget give the same selection.
- **Your tokenizer.** Plug in your model's tokenizer through `TokenCounter`. Without one, Harpe uses an estimate.
- **Confidential items stay home (Python).** With the default `destination="remote"`, items marked confidential are dropped before budgeting.
- **Whole when it fits (Python).** `route_by_fit(history, threshold_tokens)` says whether a history fits under a token threshold whole, so it can go to the model as it is, or needs retrieval and a budget. It is off unless you give a threshold.
- **Dates on remembered turns (Python).** `resolve_relative_dates(text, said_on)` writes the calendar date next to phrases such as "yesterday", "last weekend" or "three weeks ago", counted from the day the text was said, so a model that sees a remembered turn out of its conversation can still place it in time. Phrases that could mean more than one date ("in two weeks", "next Friday") are left as they are. `parse_anchor_date()` reads that day from common session timestamps.
- **Conflict notes (Python, experimental).** `find_conflicts(texts)` finds pairs where one text denies what another states, and `annotate_conflicts()` adds a note after the denial. Nothing calls it by default.
- **Small.** The Rust crate has no dependencies, and Harpe contains no network code.

## Install

This README describes 0.1.0-alpha.11 (0.1.0a11 on PyPI). It is a pre-release, so name the version.

```toml
# Rust: Cargo.toml
[dependencies]
harpe = "0.1.0-alpha.11"
```

```bash
# Python
pip install harpe==0.1.0a11
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

Try it without installing anything: [telarch.ai/harpe](https://telarch.ai/harpe) runs the crate in your browser, compiled to WebAssembly.

## Measured

Medians of five runs of [telarch-benchmarks](https://github.com/Perseus-Computing-LLC/telarch-benchmarks) v0.1.0-alpha.11, built against harpe 0.1.0-alpha.11, with synthetic data on one EC2 m8azn.metal-12xl bare-metal server. Every run, its range and the command that reproduces it are on [telarch.ai/evidence](https://telarch.ai/evidence).

All of these numbers are for the **Rust crate's greedy `LowestPriority` policy** (`knapsack_cost_curve` in telarch-benchmarks). The Python `KNAPSACK_DP` and `DUO_ATTENTION` policies solve an exact 0/1 knapsack; their kernel is timed separately below.

**Exact knapsack kernel** (`benchmarks/knapsack_dp_latency.py`, best of 5 on identical seeded instances, 13th-gen Core i5 developer container, not the EC2 harness; `benchmarks/knapsack-dp-latency-20261029.json`). With the `harpe_native` extension installed (`bindings/python`) the same DP runs in the Rust crate and chooses the same set on every instance; without it, or with `HARPE_PURE_PYTHON=1`, the Python kernel runs.

| Instance | Cells | Python kernel | Rust kernel (`harpe_native`) |
|---|---|---|---|
| 40 items, 4,000-token headroom | 160k | 1.7 ms | 0.23 ms |
| 120 items, 8,000 | 960k | 11.0 ms | 1.4 ms |
| 200 items, 16,000 | 3.2M | 36 ms | 4.8 ms |
| 250 items, 19,999 (the 5M cap) | 5.0M | 58 ms | 7.7 ms |

| Measurement (Rust, greedy `LowestPriority`) | Result |
| --- | --- |
| Fewer tokens sent over 100,000 simulated agent turns, 1,200-token budget per turn | 78.3% |
| Turns that went over the budget | 0 |
| Median call time, 10 items | 1.83 µs |
| Median call time, 200 items | 30.88 µs |
| Tokens removed to fit the budget, 200 items against 1,500 tokens | 78.0% |

"Turns that went over the budget: 0" was measured with the crate's own `HeuristicTokenCounter` counting both the items and the ceiling, so it shows the selection never exceeds the counts it is given. It says nothing about what a provider's tokenizer would charge for the same text; the heuristic undercounts emoji and hex ids and overcounts CJK and prose against tiktoken `o200k_base` (see Limits). Pinned items that alone exceed the ceiling are reported as `over_budget` rather than pruned; none of the synthetic turns hit that case.

## How it works

- **Pinned items stay.** Items that are pinned or at priority 0 (and, in Python, items of the invariant dimension) are always included. If they alone exceed the budget, nothing else is added and the report shows the overrun (`synthesize` in `src/harpe/engine.py` and `src/lib.rs`).
- **Greedy by priority by default.** `PrunePolicy.LOWEST_PRIORITY` sorts the other items by priority (lower number first; in Python, newer first among equals) and adds each one that still fits. Other policies order items by age, input order or a relevance score. The Python engine also has `KNAPSACK_DP`: an exact 0/1 knapsack on raw token counts over a per-item utility score while `items × (capacity + 1)` is at most 5,000,000 cells, and a density-greedy pass with slack fill above that; the docstring on `_solve_knapsack_dp` states the rule. An unknown policy raises `ValueError`.
- **Confidential items withheld.** With the default `destination="remote"`, the Python engine drops every item marked confidential before budgeting, so it never takes a slot or reaches the output (`src/harpe/engine.py`).
- **Token counts.** The default counter takes the larger of two estimates: the number of pieces (a run of ASCII letters, digits and underscores, or any single other non-space character, so each CJK character, emoji or accented letter is one) plus one extra per three characters of any hex run of eight or more, or a quarter of the characters. Rust and Python count the same text the same way; both test suites check the shared vectors in `tests/token_vectors.tsv`, which record tiktoken `o200k_base` counts beside the new vectors for comparison. Items that already have a `token_count` are not counted again. `harpe.tokenizers.TiktokenCounter` (install `harpe[tiktoken]`) or your own `TokenCounter` gives exact counts.
- **The report says what happened.** `SynthesisReport.effective_budget` is the ceiling the selection was made against (after any dynamic expansion), `reference_time` is the timestamp recency was scored from (the newest item, never the wall clock, so the same input gives the same output), and `over_budget`/`overflow_tokens` say when pinned items alone exceeded the ceiling. `synthesize(..., strict=True)` (Python) and `synthesize_with(..., strict: true)` (Rust) turn that case into an error.

## Limits

- **Alpha release:** APIs and structures are subject to change before 1.0.
- **Greedy is not best-fit:** The default policy can leave room that a different combination of items would fill.
- **Pinned overrun:** Pinned items are kept even when they exceed the budget; the report flags it (`over_budget`) and strict mode refuses it, but the engine does not trim pinned content.
- **Knapsack cost:** `KNAPSACK_DP` is exact only while `items × (capacity + 1)` fits in 5,000,000 cells (for example 25 items at a 200,000-token ceiling, or 1,000 items at 5,000); above that it falls back to a greedy pass whose unused capacity is below the largest single item.
- **Allocations:** Harpe allocates context items and report structures during synthesis.
- **Local token heuristic:** The default counter is an estimate, not your model's tokenizer. Against tiktoken `o200k_base` it undercounts emoji and hex ids (about 0.6× on the shared vectors) and overcounts CJK (about 1.3-1.4×) and English prose (about 1.2×, because every punctuation mark is a piece); over all 208 shared vectors it counts 1.11× what `o200k_base` does. When exact counts matter for a downstream provider, use `TiktokenCounter` or your own `TokenCounter`.
- **Not independently assessed:** No security review or certification has been completed. Designed for local and disconnected use.

## Security

Report a vulnerability to contact@perseuscomputing.com with `[SECURITY]` in the subject; please do not open a public issue. See the [security policy](https://github.com/Perseus-Computing-LLC/.github/blob/main/SECURITY.md) for response times.

## Links

- [telarch.ai/harpe](https://telarch.ai/harpe): overview and a live demo
- [Documentation (docs.rs)](https://docs.rs/harpe)
- [Changelog](CHANGELOG.md)

## License

Released under the [MIT License](LICENSE) © 2026 Perseus Computing LLC.
