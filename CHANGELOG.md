# Changelog

All notable changes to `harpe` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
