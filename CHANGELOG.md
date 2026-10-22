# Changelog

All notable changes to `harpe` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 0.1.0-alpha.7

### Added
- Python, experimental: `ProfileLedger` keeps per-attribute timelines of the user's own statements (residence,
  career, relationship, work status, health, family members, name, birthplace, birth date, gender, education)
  and resolves them by whether the attribute can change. For attributes that change, statements from the latest
  session are current and earlier ones are superseded; for attributes that do not, the first statement is the
  established record and a later claim with a different value is marked as conflicting. `route(question)` picks
  the attributes a question asks about and `render(attributes)` writes a short, dated block for the reader that
  keeps both the previous and the current state. Only the user's turns are read. `question_detail(question)`
  says which detail of a family member a question asks for. Not yet measured with a reader model.
- `ProfileLedger`: a question about the user's employment status (`question_detail` returns `"employment"`) gets a
  line stating the status the career statements imply before and after the latest change (employed, unemployed,
  self-employed or own business, retired), so a new job title or employer is not read as a change of status.
  Not yet measured with a reader model.

### Fixed
- `ProfileLedger`: birth-date statements written as "my birthdate", "my birth-date" or "my DOB" are read, so the
  first such statement becomes the established record. Two birth-date statements that give full dates in the
  same year are compared by date ("1980-03-22" and "1980-08-14" conflict; "March 22, 1980" and "22 March 1980"
  agree).
- `ProfileLedger`: "lost the job" and "lost my retail job" are read as career statements.

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
