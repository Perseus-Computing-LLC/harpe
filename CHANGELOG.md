# Changelog

All notable changes to `harpe` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- CI: every action is pinned to a full commit SHA (the Rust toolchain action followed a branch) and
  runs on Node 24. Dependabot proposes weekly grouped updates for actions, Cargo and Python
  dependencies once a release is 7 days old, and an OpenSSF Scorecard workflow reports to code
  scanning.

## [0.1.0-alpha.7] - 2026-10-23

### Changed
- Rust and Python now count tokens with one default rule: the larger of the number of pieces (words and ASCII
  punctuation marks) and a quarter of the characters, rounded up. `cargo test` and `pytest` both check the 199 shared
  vectors in `tests/token_vectors.tsv`.
  - Python's counts are unchanged for ASCII text. Characters outside ASCII, other than whitespace, now count as
    letters in Python, as they already did in Rust, so a curly quote or a full-width comma no longer splits a word.
  - Rust now counts ASCII punctuation marks as pieces, so code and other text with many of them counts higher and
    fewer such items fit a budget. The 200 items of telarch-benchmarks' phase 1 count 6,800 tokens in both languages
    (Rust counted 4,900), and Rust now keeps 44 of them within the 1,500-token budget instead of 61.
- Python: without pruning hooks, `synthesize()` copies only the items it has to count. Items that already have a
  `token_count` go through as they are, so the report holds the caller's own objects for them. Hooks may change
  items in place, so with hooks every item is still copied.

## [0.1.0-alpha.6] - 2026-10-21

### Added
- Python: `resolve_relative_dates(text, anchor)` adds the calendar date to relative time expressions, counted
  from the date the text was said: "yesterday" said on 11 May 2023 becomes "yesterday [10 May 2023]";
  "last weekend", "three weeks ago" and "last month" gain their dates. Expressions with more than one
  reading ("in two weeks", "next ...") are left alone. `parse_anchor_date(value)` reads the date in session
  timestamps such as "1:56 pm on 8 May, 2023", "2023/05/20 (Sat) 02:21" or "2024-03-15". Apply it to the text
  a model reads, not to what is indexed. In telarch-benchmarks (gpt-4o-mini reader) it raised LoCoMo by 4.5
  points, 20.3 on its temporal questions; LongMemEval-S and BEAM did not change.
- Python, experimental: `find_conflicts(texts, query=None, ...)` finds pairs where one text denies what another
  states ("I've never written integration tests" against "I've written integration tests ...");
  `conflict_note()` and `annotate_conflicts()` put a note after the denying sentence. In telarch-benchmarks the
  notes did not improve BEAM's contradiction questions (slightly lower at 1M), so nothing calls it by default.

## [0.1.0-alpha.5] - 2026-10-21

### Added
- Python: fit-aware routing. `route_by_fit(items, threshold_tokens, token_counter=None)`, and
  `ContextBudgetEngine.route(items, threshold_tokens)` with the engine's counter, return a `RouteDecision`:
  `ContextRoute.WHOLE` with the history's token count when every item fits under the threshold (hand the
  history over in order, unpruned), else `ContextRoute.RETRIEVE`. Counting stops at the first item over
  the threshold, items with a `token_count` keep it, and a threshold of 0 or `None` turns routing off.
  telarch-benchmarks measured a threshold of 32,000 reader tokens.

## [0.1.0-alpha.4] - 2026-10-19

### Fixed
- Version strings agree: `pyproject.toml` said 0.1.0a3 while `harpe.__version__` said 0.1.0a4.
- README claims checked against the code. Removed: the measured-figures table (kept in
  telarch-benchmarks), "strict token ceiling" (pinned items can exceed it), and the statement that
  Harpe does no dynamic programming (the Python engine has `PrunePolicy.KNAPSACK_DP`).
- SECURITY.md states the scope: Harpe has no network code, and pinned items can exceed the budget
  (check `consumed_tokens`).

### Added
- Release workflow: builds and attests the sdist, wheel and crate, installs the wheel with its
  dependencies on Ubuntu and macOS arm64, creates a GitHub release naming the commit, and publishes
  to PyPI and crates.io by trusted publishing from protected environments.
- CI runs the Python tests on macOS arm64 as well as Ubuntu, and installs the built wheel in a clean
  virtual environment.

## [0.1.0-alpha.3] - 2026-10-18

### Added
- Dynamic Tool Output Compression (DTOC) support in Python engine.
- Heuristic token counter with ceiling arithmetic.
- Support for confidential item withholding on remote model destinations.
- MSRV enforcement for `rust-version = "1.85"` in `Cargo.toml`.
- Reproducible benchmark integration and claims alignment with `telarch-benchmarks`.

### Changed
- Replaced manual saturation arithmetic with `saturating_sub` in `available_context()`.
- Standardized clippy warnings with zero warnings under `-D warnings`.
- Updated package metadata to reflect Telarch naming and official documentation links.

## [0.1.0-alpha.2] - 2026-10-13

### Added
- Initial public release of Harpe context synthesizer crate on crates.io.
- Python bindings and Pydantic data models on PyPI.
- Heuristic token estimation and priority-ordered greedy pruning policies (`LowestPriority`, `Fifo`, `TruncateTail`).
