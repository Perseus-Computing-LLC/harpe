//! # harpe
//!
//! Deterministic context synthesis and token budgeting for autonomous AI agents.
//!
//! Harpe provides deterministic, microsecond token budgeting and priority-based
//! context trimming for agentic loops. When language models run multi-turn agent
//! workflows, context windows can quickly exceed hardware or inference limits.
//! Uncontrolled growth results in context truncation, lost system prompts, or
//! excessive inference latency.
//!
//! Harpe solves this by calculating available context headroom, prioritizing
//! pinned instructions, and pruning lower-priority context items before model
//! invocation.
//!
//! ## Architecture Overview
//!
//! 1. **Token Accounting**: [`TokenBudget`] defines the maximum context envelope,
//!    deducting reserved tokens for model output, tool specifications, and safety margin.
//! 2. **Context Entities**: [`ContextItem`] represents individual context components
//!    (system prompts, active tool outputs, conversation turns) tagged with numeric priorities.
//! 3. **Synthesis Engine**: [`ContextBudgetEngine`] resolves token counts and applies
//!    pruning strategies such as [`PrunePolicy::LowestPriority`] or [`PrunePolicy::Fifo`]
//!    to produce an exact [`SynthesisReport`] fitting within the budget.
//!
//! ## Quick Start
//!
//! ```rust
//! use harpe::{ContextBudgetEngine, ContextItem, ContextSynthesizer, PrunePolicy, TokenBudget};
//!
//! // Define a total context envelope of 4096 tokens, reserving 1024 for output and tools
//! let mut budget = TokenBudget::new(4096);
//! budget.reserved_completion = 512;
//! budget.reserved_tools = 256;
//! budget.safety_margin = 256;
//!
//! assert_eq!(budget.available_context(), 3072);
//!
//! // Prepare context items with explicit priorities (0 is pinned/highest priority)
//! let system_prompt = ContextItem::new("sys", "You are an autonomous systems assistant.", 0);
//! let conversation = ContextItem::new("hist", "Previous user query and reasoning traces.", 50);
//! let scratchpad = ContextItem::new("scratch", "Temporary debug buffer with logs.", 90);
//!
//! let engine = ContextBudgetEngine::default();
//! let report = engine.synthesize(
//!     &[system_prompt, conversation, scratchpad],
//!     &budget,
//!     PrunePolicy::LowestPriority,
//! ).expect("Synthesis should succeed");
//!
//! println!("Consumed tokens: {}", report.consumed_tokens);
//! println!("Remaining tokens: {}", report.budget_remaining);
//! ```

pub mod state_ledger;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Pruning policies supported by the context synthesis engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrunePolicy {
    /// First-In First-Out: Prune older context items first based on `created_at` timestamp.
    Fifo,
    /// Priority-based: Retain items with lower priority numbers first (0 = highest priority).
    LowestPriority,
    /// Drop tail items that exceed available capacity in sequential order.
    TruncateTail,
    /// Discard items considered least relevant to the current objective.
    LeastRelevant,
}

/// Token budget configuration for an agent invocation turn.
///
/// Tracks the total available model context window and subtracts necessary reserves
/// for generation and tools to calculate safe available context headroom.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenBudget {
    /// Total token limit of the target model (for example, 8192 or 128000).
    pub total_budget: usize,
    /// Tokens reserved strictly for the model response generation.
    pub reserved_completion: usize,
    /// Tokens reserved for JSON schemas and tool definitions.
    pub reserved_tools: usize,
    /// Safety headroom buffer to absorb tokenizer discrepancies.
    pub safety_margin: usize,
}

impl TokenBudget {
    /// Create a new budget with the specified total token capacity.
    pub fn new(total_budget: usize) -> Self {
        Self {
            total_budget,
            reserved_completion: 0,
            reserved_tools: 0,
            safety_margin: 0,
        }
    }

    /// Calculate the usable context token capacity after all reserves are deducted.
    pub fn available_context(&self) -> usize {
        let reservations = self.reserved_completion + self.reserved_tools + self.safety_margin;
        self.total_budget.saturating_sub(reservations)
    }

    /// Returns true if the provided token usage exceeds available context capacity.
    pub fn is_exceeded(&self, used_tokens: usize) -> bool {
        used_tokens > self.available_context()
    }

    /// Calculate context utilization percentage.
    pub fn utilization_pct(&self, used_tokens: usize) -> f64 {
        let avail = self.available_context();
        if avail == 0 {
            100.0
        } else {
            (used_tokens as f64 / avail as f64) * 100.0
        }
    }
}

/// A discreet unit of context presented to the synthesis engine.
#[derive(Debug, Clone, PartialEq)]
pub struct ContextItem {
    /// Unique identifier for this context entry.
    pub id: String,
    /// Logical role (for example: system, user, assistant, context).
    pub role: String,
    /// Raw textual content.
    pub content: String,
    /// Measured token count (computed dynamically if initialized as 0).
    pub token_count: usize,
    /// Retention priority. Lower values indicate higher importance. Priority 0 is pinned.
    pub priority: u8,
    /// If true, this item will never be pruned regardless of budget pressure.
    pub pinned: bool,
    /// Timestamp (UNIX epoch in seconds or fractional seconds) for FIFO ordering.
    pub created_at: f64,
}

impl ContextItem {
    /// Create a new context item with an identifier, content, and priority.
    /// Priority 0 automatically marks the item as pinned.
    pub fn new(id: impl Into<String>, content: impl Into<String>, priority: u8) -> Self {
        Self {
            id: id.into(),
            role: "context".to_string(),
            content: content.into(),
            token_count: 0,
            priority,
            pinned: priority == 0,
            created_at: 0.0,
        }
    }
}

/// Detailed outcome of a context synthesis execution.
#[derive(Debug, Clone, PartialEq)]
pub struct SynthesisReport {
    /// Context items selected to fit within the available budget.
    pub included_items: Vec<ContextItem>,
    /// Context items pruned due to budget constraints.
    pub pruned_items: Vec<ContextItem>,
    /// Total tokens consumed by included items.
    pub consumed_tokens: usize,
    /// Unallocated tokens remaining in available context space.
    pub budget_remaining: usize,
    /// Total tokens discarded across all pruned items.
    pub pruned_tokens: usize,
    /// The pruning strategy that was applied during synthesis.
    pub policy_applied: PrunePolicy,
}

/// Errors returned during context synthesis and budgeting operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarpeError {
    /// Pinned items alone exceed the entire available context budget.
    BudgetOverflow(String),
    /// Internal failure during context synthesis or pruning calculation.
    SynthesisFailed(String),
}

impl std::fmt::Display for HarpeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BudgetOverflow(msg) => write!(f, "Budget overflow: {}", msg),
            Self::SynthesisFailed(msg) => write!(f, "Synthesis failed: {}", msg),
        }
    }
}

impl std::error::Error for HarpeError {}

/// Trait for measuring token counts of text slices.
pub trait TokenCounter: Send + Sync {
    /// Return the measured or estimated token count for the given text.
    fn count_tokens(&self, text: &str) -> usize;
}

/// Hook for lifecycle interception before and after pruning runs.
pub trait PruningHook: Send + Sync {
    /// Invoked before candidate pruning begins.
    fn pre_prune(&self, items: &[ContextItem]);
    /// Invoked after synthesis completes with the finalized report.
    fn post_prune(&self, report: &SynthesisReport);
}

/// Trait implemented by context synthesis engines.
pub trait ContextSynthesizer: Send + Sync {
    /// Synthesize and prune context items according to the supplied budget and policy.
    fn synthesize(
        &self,
        items: &[ContextItem],
        budget: &TokenBudget,
        policy: PrunePolicy,
    ) -> Result<SynthesisReport, HarpeError>;
}

/// Default token estimate, the same in Rust and Python: the larger of (pieces + hex extras) and a
/// quarter of the characters, rounded up. A piece is a run of ASCII letters, digits and underscores,
/// or any single other non-whitespace character: an ASCII symbol or any non-ASCII code point, so
/// CJK text, emoji and accented letters count one per code point instead of joining a word. Every
/// maximal run of 8 or more hex digits adds `len / 3` extras on top of the piece it sits in, so
/// hashes and ids cost about one token per three characters. Whitespace is what Python's
/// `str.isspace()` accepts: Unicode `White_Space` plus U+001C to U+001F. Both test suites check
/// `tests/token_vectors.tsv`. It is a floor, not a tokenizer: pass your model's [`TokenCounter`]
/// for exact counts.
#[derive(Debug, Default, Clone)]
pub struct HeuristicTokenCounter;

impl HeuristicTokenCounter {
    // How each ASCII character counts. Any other character is a space if it is Unicode
    // `White_Space`, and otherwise a symbol: one piece on its own that ends any word run.
    const WORD: u8 = 0;
    const SPACE: u8 = 1;
    const SYMBOL: u8 = 2;
    // Runs of at least this many hex digits get the extra per-3-characters charge.
    const HEX_RUN_MIN: usize = 8;
    const ASCII_CLASS: [u8; 128] = {
        let mut class = [Self::SYMBOL; 128];
        let mut b = 0;
        while b < 128 {
            let c = b as u8;
            if c.is_ascii_alphanumeric() || c == b'_' {
                class[b] = Self::WORD;
            } else if matches!(c, b'\t'..=b'\r' | 0x1c..=b' ') {
                class[b] = Self::SPACE;
            }
            b += 1;
        }
        class
    };
}

impl TokenCounter for HeuristicTokenCounter {
    fn count_tokens(&self, text: &str) -> usize {
        let (mut pieces, mut chars, mut in_word, mut hex_run) = (0usize, 0usize, false, 0usize);
        for c in text.chars() {
            // Hex runs are tracked across piece boundaries exactly like Python's `[0-9a-fA-F]{8,}`:
            // a maximal run inside a longer word still counts.
            if c.is_ascii_hexdigit() {
                hex_run += 1;
            } else {
                if hex_run >= Self::HEX_RUN_MIN {
                    pieces += hex_run / 3;
                }
                hex_run = 0;
            }
            let class = if c.is_ascii() {
                Self::ASCII_CLASS[c as usize]
            } else if c.is_whitespace() {
                Self::SPACE
            } else {
                Self::SYMBOL
            };
            let starts_piece = class == Self::SYMBOL || (class == Self::WORD && !in_word);
            pieces += usize::from(starts_piece);
            chars += 1;
            in_word = class == Self::WORD;
        }
        if hex_run >= Self::HEX_RUN_MIN {
            pieces += hex_run / 3;
        }
        pieces.max(chars.div_ceil(4))
    }
}

/// Primary context synthesis and budgeting engine.
pub struct ContextBudgetEngine {
    counter: Box<dyn TokenCounter>,
}

impl Default for ContextBudgetEngine {
    fn default() -> Self {
        Self {
            counter: Box::new(HeuristicTokenCounter),
        }
    }
}

impl ContextBudgetEngine {
    /// Instantiate a budget engine with a custom token counter implementation.
    pub fn new(counter: Box<dyn TokenCounter>) -> Self {
        Self { counter }
    }
}

impl ContextSynthesizer for ContextBudgetEngine {
    fn synthesize(
        &self,
        items: &[ContextItem],
        budget: &TokenBudget,
        policy: PrunePolicy,
    ) -> Result<SynthesisReport, HarpeError> {
        let mut resolved: Vec<ContextItem> = Vec::with_capacity(items.len());
        for it in items {
            let mut clone = it.clone();
            if clone.token_count == 0 {
                clone.token_count = self.counter.count_tokens(&clone.content);
            }
            resolved.push(clone);
        }

        let capacity = budget.available_context();
        let mut pinned = Vec::new();
        let mut unpinned = Vec::new();

        for it in resolved.into_iter() {
            if it.pinned || it.priority == 0 {
                pinned.push(it);
            } else {
                unpinned.push(it);
            }
        }

        let pinned_tokens: usize = pinned.iter().map(|it| it.token_count).sum();

        if pinned_tokens >= capacity {
            let pruned_tokens = unpinned.iter().map(|it| it.token_count).sum();
            return Ok(SynthesisReport {
                included_items: pinned,
                pruned_items: unpinned,
                consumed_tokens: pinned_tokens,
                budget_remaining: 0,
                pruned_tokens,
                policy_applied: policy,
            });
        }

        let mut remaining = capacity - pinned_tokens;
        let mut included_unpinned = Vec::new();
        let mut pruned_unpinned = Vec::new();

        match policy {
            PrunePolicy::LowestPriority => {
                let mut sorted = unpinned;
                sorted.sort_by_key(|a| a.priority);
                for it in sorted {
                    if it.token_count <= remaining {
                        remaining -= it.token_count;
                        included_unpinned.push(it);
                    } else {
                        pruned_unpinned.push(it);
                    }
                }
            }
            PrunePolicy::Fifo => {
                let mut sorted = unpinned;
                sorted.sort_by(|a, b| {
                    b.created_at
                        .partial_cmp(&a.created_at)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                for it in sorted {
                    if it.token_count <= remaining {
                        remaining -= it.token_count;
                        included_unpinned.push(it);
                    } else {
                        pruned_unpinned.push(it);
                    }
                }
            }
            PrunePolicy::TruncateTail | PrunePolicy::LeastRelevant => {
                for it in unpinned {
                    if it.token_count <= remaining {
                        remaining -= it.token_count;
                        included_unpinned.push(it);
                    } else {
                        pruned_unpinned.push(it);
                    }
                }
            }
        }

        let consumed: usize = pinned
            .iter()
            .chain(included_unpinned.iter())
            .map(|it| it.token_count)
            .sum();
        let pruned: usize = pruned_unpinned.iter().map(|it| it.token_count).sum();

        let mut final_included = pinned;
        final_included.extend(included_unpinned);

        Ok(SynthesisReport {
            included_items: final_included,
            pruned_items: pruned_unpinned,
            consumed_tokens: consumed,
            budget_remaining: capacity.saturating_sub(consumed),
            pruned_tokens: pruned,
            policy_applied: policy,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_budget_available_context() {
        let mut budget = TokenBudget::new(5000);
        budget.reserved_completion = 1000;
        budget.reserved_tools = 500;
        budget.safety_margin = 200;

        assert_eq!(budget.available_context(), 3300);
        assert!(!budget.is_exceeded(3300));
        assert!(budget.is_exceeded(3301));
    }

    #[test]
    fn test_heuristic_token_counter() {
        let counter = HeuristicTokenCounter;
        assert_eq!(counter.count_tokens(""), 0);
        assert!(counter.count_tokens("Hello world from Harpe") >= 4);
    }

    #[test]
    fn test_heuristic_token_counter_matches_the_shared_vectors() {
        let counter = HeuristicTokenCounter;
        let mut checked = 0;
        for line in include_str!("../tests/token_vectors.tsv").lines() {
            if line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').collect();
            assert_eq!(fields.len(), 3, "{line}");
            let bytes: Vec<u8> = (0..fields[1].len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&fields[1][i..i + 2], 16).unwrap())
                .collect();
            let text = String::from_utf8(bytes).unwrap();
            assert_eq!(
                counter.count_tokens(&text),
                fields[0].parse::<usize>().unwrap(),
                "{}",
                fields[2]
            );
            checked += 1;
        }
        assert_eq!(checked, 208);
    }

    #[test]
    fn test_lowest_priority_pruning() {
        let engine = ContextBudgetEngine::default();
        let budget = TokenBudget::new(100);

        let items = vec![
            ContextItem {
                id: "pin".into(),
                role: "system".into(),
                content: "Keep this".into(),
                token_count: 20,
                priority: 0,
                pinned: true,
                created_at: 1.0,
            },
            ContextItem {
                id: "high".into(),
                role: "user".into(),
                content: "Important message".into(),
                token_count: 30,
                priority: 10,
                pinned: false,
                created_at: 2.0,
            },
            ContextItem {
                id: "low".into(),
                role: "context".into(),
                content: "Low importance".into(),
                token_count: 60,
                priority: 80,
                pinned: false,
                created_at: 3.0,
            },
        ];

        let report = engine
            .synthesize(&items, &budget, PrunePolicy::LowestPriority)
            .unwrap();
        assert_eq!(report.included_items.len(), 2);
        assert_eq!(report.pruned_items.len(), 1);
        assert_eq!(report.pruned_items[0].id, "low");
        assert_eq!(report.consumed_tokens, 50);
        assert_eq!(report.budget_remaining, 50);
    }
}
