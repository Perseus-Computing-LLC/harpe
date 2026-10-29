# Changelog

All notable changes to `harpe` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- H-R3: typed sidecar for `confidential` and `scope` flags on DTOC compressed blocks. Artifacts and
  placeholder headers preserve `confidential=true` and `scope=<scope>`; `to_context_item()` converts
  artifacts to `ContextItem` with matching sensitivity and scope metadata, surviving compaction round-trips.
- H-R4: killed-process (exit 143 / SIGTERM) outputs automatically marked `partial=true` and annotated with
  truncation/partial preview headers. Exit code and truncation markers preserved through DTOC placeholders.
- H-R2: role-typed serialization. `SynthesisReport.to_messages()` returns one `{role, id, content}` per
  included item, in order, never merged, so a tool output that contains a forged `user:` or `system:` turn
  stays a tool message. `to_context_str()` / `render()` keep the old join when every item shares one role; with
  mixed roles they now fence each item in a `<harpe-role name=... id=... fence=NONCE>` block under a nonce drawn
  per render (content produced before the render cannot forge the closing tag), raise `MixedRoleRenderError`
  with `mixed_roles="error"`, or flatten only with `mixed_roles="flatten"`. `fence_nonce=` fixes the nonce
  for a reproducible render.

### Changed
- H-E: unexport `BehavioralGovernor`, `GovernorDecision`, `GovernorVerdict` from `harpe` root.
  Decided: stop exporting as a safety control until backed by a formal policy engine; documented
  re-entry criteria in `src/harpe/governor.py`. Modules remain accessible via `from harpe.governor import ...`.
- `ContextItem.role` is validated: `system`, `user`, `assistant`, `tool` or `context`. A role that is not one of
  these cannot be serialized faithfully, so it is refused at construction.
- H-B: the exact 0/1 knapsack behind `KNAPSACK_DP` and `DUO_ATTENTION` is about 2.5x faster with identical
  selections. It still evaluates the textbook one-row recurrence in input order with the same strict-improvement
  tie rule, but no longer visits every cell one at a time: items heavier than the capacity are rejected before
  the table is built, weights and capacity are divided by their common divisor when there is one (lossless),
  cells outside the reachable prefix or the backtrace suffix are skipped, and because the row is monotone two
  comparisons per block decide whether a block of cells changes entirely (filled by a slice) or not at all,
  with only the mixed blocks scanned cell by cell. 200 items of 20-400 tokens at an 8,000-token ceiling went
  from 52 ms to 20.5 ms per `synthesize` (Python 3.14, same host); `tests/test_knapsack_fast_hb.py` checks
  selection equality against a two-dimensional reference table on 120 seeded instances, including ties, pinned
  items, oversize items and shared divisors, and holds the timing under 26 ms.

### Fixed
- `ProfileLedger`: a kinship noun owned by a named third party or by "its" ("Claire's father lives in Boston now",
  "its mother") is read as theirs, the same as "her father" already was, whatever came earlier in the sentence.
  Before, `_about_someone_else` only knew the pronoun forms, so such a sentence could be recorded as the user's
  own residence, career or parent.
- `ProfileLedger`: a singular sibling denial ("I don't have a sister", "I never had a brother or a sister") is
  read as has:no. Before, only the plural forms were, so the sentence was read as one sister (has:yes, count:1)
  and rendered as matching an earlier "my sister ..." record instead of as a later conflicting claim. A denial
  of one kind next to a count of another ("I don't have a sister, but I have two brothers") still reads as
  has:yes with that count. One MemConflict replay question moves accordingly (listed in the test).

## [0.1.0-alpha.10] - 2026-10-28

Repair set from the 2026-10-28 review. Each item is one commit on `fix/review-2026-10-28` and ships with a regression
test. alpha.9 (below) was never tagged or published; its changes ship here, and alpha.8 is the previous release
on crates.io and PyPI.

### Fixed

- **H-01** `synthesize` rebuilt the included/pruned lists by item id, so two items sharing an id were both emitted when
  either was selected (100 + 1,000 tokens under a 500 ceiling gave `consumed_tokens=1100`). Selection is tracked by
  object identity everywhere, including the `DUO_ATTENTION` sink/local split, duplicate ids raise `ValueError` up
  front, and the dead `final_unpinned_in_order` is gone.
- **H-02** `_solve_knapsack_dp` quantised weights into 1,000 bins, so at a 200,000-token ceiling every item cost at
  least 200 tokens and 1,500 one-token items left 99.5% of the budget unused. It is now an exact 0/1 knapsack on raw
  token weights (1-D value row, `bytearray` parent rows) while `n × (capacity + 1) <= 5,000,000`, and a density-greedy
  pass with slack fill above that. Direct solver tests plus `KNAPSACK_DP`/`DUO_ATTENTION` policy tests.
- **H-03** `HeuristicTokenCounter` treated every non-ASCII character as a word letter. Each non-ASCII code point is
  now one piece, runs of 8+ hex digits add one per three characters, and the `ceil(len/4)` floor stays; applied
  identically in Rust. `tests/token_vectors.tsv` gains CJK, emoji, sha256, URL, UUID and commit-hash vectors with
  the tiktoken `o200k_base` counts recorded beside them. New `harpe.tokenizers.TiktokenCounter` behind the
  `[tiktoken]` extra.
- **H-04** `resolve_dynamic_budget` capped the widened budget at an absolute 4,000 tokens, shrinking any budget above
  that while the report said "expanded". The cap is a factor (`max_factor`; engine kwarg `max_dynamic_factor`)
  clamped to `>= 1.0`, and `SynthesisReport.effective_budget` records the ceiling used on every path.
- **M-01** When pinned, priority-0 or invariant items alone exceed the ceiling, `SynthesisReport.over_budget` and
  `overflow_tokens` say so (Python and Rust). `synthesize(strict=True)` raises `BudgetOverflowError` (Python);
  `synthesize_with(.., strict: true)` returns `HarpeError::BudgetOverflow` (Rust).
- **M-02** `KNAPSACK_DP` and `DUO_ATTENTION` scored recency against `time.time()`, so the same inputs could select
  differently hours apart. `synthesize(now=...)` is threaded to `_compute_item_utility`; the default is the newest
  `created_at` among the items and is reported as `SynthesisReport.reference_time`.
- **M-03** `BehavioralGovernor`: a revoked invariant without a `key` suppressed every action (empty string is a
  substring of everything); only a non-empty stripped key matches. Blocked patterns collapse whitespace and accept
  the recursive flag in any group or order (`rm -r -f /`, `rm -fr ~`, `rm -rf *`, `--recursive --force`), and the
  docstring says they are a last-resort heuristic, not a sandbox.
- **M-04** Rust `content_words` split on every non-lowercase character and produced `tests`, `socket`, `events` and
  `caf` where Python's `\b[a-z]{3,}\b` produces nothing. It now matches; `tests/content_word_vectors.tsv` is
  checked by both suites.
- **M-05** `DynamicToolCompressor` kept every artifact for the life of the process. The store is an
  `LRUArtifactStore` capped by raw bytes (`store_max_bytes`, default 64 MiB); evicted refs restore as `None`.

### Changed

- **R-1** Policy selection moved out of `synthesize` into `_select_unpinned(policy, items, capacity, ...)`; the
  `synthesize` body is 75 lines, and an unknown policy raises `ValueError` instead of running `LOWEST_PRIORITY`.
- README separates the Rust greedy latency numbers from the Python knapsack and says the "zero overruns" figure was
  measured with the crate's own `HeuristicTokenCounter`.
- `pyproject.toml`: optional extra `tiktoken`.


## [0.1.0-alpha.9] - 2026-10-27

Verified offline against the banked MemConflict Round 5 corpus (3,750 questions, `tests/test_memconflict_replay.py`):
34 questions change, all static-conflict family questions whose gold answer says the sources disagree; the other
3,716 ledger blocks are byte-identical to 0.1.0a8. No paid run.

### Added
- Python and Rust: `StateLedger` programmatic API. `Claim.seq` (insertion order; `(date, seq)` is time order, unit
  ids are never sorted as strings), `StatePair.strength` (length-normalised overlap, `|shared|^2 / sqrt(|a|*|d|)`,
  so a tight polarity flip outranks a long request sentence), `conflicts_for_entity(terms, since=, until=,
  min_shared=, limit=)`, `timeline(terms)` and `latest_state(terms)`. `conflicts_for(question)` (the prompt path)
  ranks by the same strength; its gate and header text are unchanged. Rust mirror `harpe::state_ledger` is
  dependency-free and passes the same fixtures.

### Fixed
- `ProfileLedger`: a kinship noun owned by someone else ("her father has alternate weekends", "shared only with
  the father" of a child) is no longer read as the user's own family statement, so it cannot become the
  established record. "my dad" is always the user's.
- `ProfileLedger`: a later "I have no X" after an inferred "has X" is marked `LATER CONFLICTING CLAIM`. Negations
  the previous patterns did not read: "don't really have siblings", "no known / recorded / biological mother",
  "no maternal records", "she passed away", "raised solely by my father", "the records list the mother as
  unknown", "raised in an orphanage". A reported or denied negative ("a friend assumed I had no siblings, which
  isn't true") is not the user's own "no".

## [0.1.0-alpha.8] - 2026-10-27


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
