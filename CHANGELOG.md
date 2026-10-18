# Changelog

All notable changes to `harpe` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
