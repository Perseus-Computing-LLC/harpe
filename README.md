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

This README describes 0.1.0-alpha.6 (0.1.0a6 on PyPI). It is a pre-release, so name the version.

```toml
# Rust: Cargo.toml
[dependencies]
harpe = "0.1.0-alpha.6"
```

```bash
# Python
pip install harpe==0.1.0a6
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

Medians of five runs of [telarch-benchmarks](https://github.com/Perseus-Computing-LLC/telarch-benchmarks) v0.1.0-alpha.7, built against harpe 0.1.0-alpha.6, with synthetic data on one EC2 m8azn.metal-12xl bare-metal server. Every run, its range and the command that reproduces it are on [telarch.ai/evidence](https://telarch.ai/evidence).

| Measurement | Result |
| --- | --- |
| Fewer tokens sent over 100,000 simulated agent turns, 1,200-token budget per turn | 78.3% |
| Turns that went over the budget | 0 |
| Median call time, 10 items | 1.61 µs |
| Median call time, 200 items | 26.03 µs |
| Tokens removed to fit the budget, 200 items against 1,500 tokens | 69.5% |

## How it works

- **Pinned items stay.** Items that are pinned or at priority 0 (and, in Python, items of the invariant dimension) are always included. If they alone exceed the budget, nothing else is added and the report shows the overrun (`synthesize` in `src/harpe/engine.py` and `src/lib.rs`).
- **Greedy by priority by default.** `PrunePolicy.LOWEST_PRIORITY` sorts the other items by priority (lower number first; in Python, newer first among equals) and adds each one that still fits. Other policies order items by age, input order or a relevance score; the Python engine also has a dynamic-programming selection over a per-item utility score (`KNAPSACK_DP`).
- **Confidential items withheld.** With the default `destination="remote"`, the Python engine drops every item marked confidential before budgeting, so it never takes a slot or reaches the output (`src/harpe/engine.py`).
- **Token counts.** The default counter takes the larger of two estimates: the number of words and ASCII punctuation marks, or a quarter of the characters. Rust and Python count the same text the same way; both test suites check the shared vectors in `tests/token_vectors.tsv`. Items that already have a `token_count` are not counted again. Pass your own `TokenCounter` for exact counts.

## Limits

- **Alpha release:** APIs and structures are subject to change before 1.0.
- **Greedy is not best-fit:** The default policy can leave room that a different combination of items would fill.
- **Pinned overrun:** Pinned items are kept even when they exceed the budget.
- **Time-dependent policies:** `KNAPSACK_DP` and `DUO_ATTENTION` score items by age against the current time, so their selection can change as time passes.
- **Allocations:** Harpe allocates context items and report structures during synthesis.
- **Local token heuristic:** The default counter is an estimate, not your model's tokenizer. It treats characters outside ASCII, other than spaces, as letters, so it can undercount text written without spaces, such as Chinese or Japanese. When exact counts matter for a downstream provider, pass your own `TokenCounter`.
- **Not independently assessed:** No security review or certification has been completed. Designed for local and disconnected use.

## Security

Report a vulnerability to contact@perseuscomputing.com with `[SECURITY]` in the subject; please do not open a public issue. See the [security policy](https://github.com/Perseus-Computing-LLC/.github/blob/main/SECURITY.md) for response times.

## Links

- [telarch.ai/harpe](https://telarch.ai/harpe): overview and a live demo
- [Documentation (docs.rs)](https://docs.rs/harpe)
- [Changelog](CHANGELOG.md)

## License

Released under the [MIT License](LICENSE) © 2026 Perseus Computing LLC.
