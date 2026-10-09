//! # harpe
//!
//! Deterministic context synthesis and token budgeting for cognitive AI agents.
//! Clean-room implementation © 2026 Perseus Computing LLC.

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrunePolicy {
    Fifo,
    LowestPriority,
    TruncateTail,
    LeastRelevant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenBudget {
    pub total_budget: usize,
    pub reserved_completion: usize,
    pub reserved_tools: usize,
    pub safety_margin: usize,
}

impl TokenBudget {
    pub fn new(total_budget: usize) -> Self {
        Self {
            total_budget,
            reserved_completion: 0,
            reserved_tools: 0,
            safety_margin: 0,
        }
    }

    pub fn available_context(&self) -> usize {
        let reservations = self.reserved_completion + self.reserved_tools + self.safety_margin;
        if self.total_budget > reservations {
            self.total_budget - reservations
        } else {
            0
        }
    }

    pub fn is_exceeded(&self, used_tokens: usize) -> bool {
        used_tokens > self.available_context()
    }

    pub fn utilization_pct(&self, used_tokens: usize) -> f64 {
        let avail = self.available_context();
        if avail == 0 {
            100.0
        } else {
            (used_tokens as f64 / avail as f64) * 100.0
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ContextItem {
    pub id: String,
    pub role: String,
    pub content: String,
    pub token_count: usize,
    pub priority: u8, // 0 = pinned/critical, 100 = discardable
    pub pinned: bool,
    pub created_at: f64,
}

impl ContextItem {
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

#[derive(Debug, Clone, PartialEq)]
pub struct SynthesisReport {
    pub included_items: Vec<ContextItem>,
    pub pruned_items: Vec<ContextItem>,
    pub consumed_tokens: usize,
    pub budget_remaining: usize,
    pub pruned_tokens: usize,
    pub policy_applied: PrunePolicy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HarpeError {
    BudgetOverflow(String),
    SynthesisFailed(String),
}

pub trait TokenCounter: Send + Sync {
    fn count_tokens(&self, text: &str) -> usize;
}

pub trait PruningHook: Send + Sync {
    fn pre_prune(&self, items: &[ContextItem]);
    fn post_prune(&self, report: &SynthesisReport);
}

pub trait ContextSynthesizer: Send + Sync {
    fn synthesize(
        &self,
        items: &[ContextItem],
        budget: &TokenBudget,
        policy: PrunePolicy,
    ) -> Result<SynthesisReport, HarpeError>;
}

#[derive(Debug, Default, Clone)]
pub struct HeuristicTokenCounter;

impl TokenCounter for HeuristicTokenCounter {
    fn count_tokens(&self, text: &str) -> usize {
        if text.is_empty() {
            return 0;
        }
        let words = text.split_whitespace().count();
        let chars = text.chars().count();
        let char_estimate = (chars + 3) / 4;
        std::cmp::max(words, char_estimate)
    }
}

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
                sorted.sort_by(|a, b| a.priority.cmp(&b.priority));
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
