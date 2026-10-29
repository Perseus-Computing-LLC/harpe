//! Exact 0/1 knapsack kernel behind the Python engine's `PrunePolicy::KNAPSACK_DP` (issue #11).
//!
//! `exact_knapsack` is the textbook one-row DP, `dp[c] = max(dp[c], dp[c - w] + u)`, run for every item
//! in input order and every cell `c` from `capacity` down to `w`, with an item recorded as taken at `c`
//! only when it strictly improves the cell. That is the rule the Python kernel
//! (`ContextBudgetEngine._exact_knapsack`) has always used, so ties resolve the same way (earlier items
//! win) and the float additions happen in the same order on the same `f64` values. The two kernels are
//! held to identical selections by `tests/test_native_knapsack.py` in the Python package.
//!
//! Memory: one byte of keep flags per (item, cell), `n * (capacity + 1)` bytes, as in Python.

/// Error returned for input the kernel is not defined on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KnapsackError {
    /// `weights` and `utilities` differ in length.
    LengthMismatch { weights: usize, utilities: usize },
    /// A weight is outside `1..=capacity`; the caller filters those out first.
    WeightOutOfRange {
        index: usize,
        weight: usize,
        capacity: usize,
    },
}

impl std::fmt::Display for KnapsackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KnapsackError::LengthMismatch { weights, utilities } => {
                write!(
                    f,
                    "weights ({weights}) and utilities ({utilities}) differ in length"
                )
            }
            KnapsackError::WeightOutOfRange {
                index,
                weight,
                capacity,
            } => {
                write!(
                    f,
                    "weight {weight} at index {index} is outside 1..={capacity}"
                )
            }
        }
    }
}

impl std::error::Error for KnapsackError {}

/// Indices (ascending) of the items of maximum total utility whose weights sum to at most `capacity`.
///
/// Every weight must lie in `1..=capacity`; utilities may be any finite `f64`. Ties resolve to earlier
/// items exactly as the Python kernel resolves them.
pub fn exact_knapsack(
    weights: &[usize],
    utilities: &[f64],
    capacity: usize,
) -> Result<Vec<usize>, KnapsackError> {
    if weights.len() != utilities.len() {
        return Err(KnapsackError::LengthMismatch {
            weights: weights.len(),
            utilities: utilities.len(),
        });
    }
    for (index, &weight) in weights.iter().enumerate() {
        if weight == 0 || weight > capacity {
            return Err(KnapsackError::WeightOutOfRange {
                index,
                weight,
                capacity,
            });
        }
    }
    let n = weights.len();
    if n == 0 {
        return Ok(Vec::new());
    }
    let cells = capacity + 1;
    let mut dp = vec![0.0f64; cells];
    let mut keep = vec![0u8; n * cells];
    for (i, (&w, &u)) in weights.iter().zip(utilities).enumerate() {
        let row = &mut keep[i * cells..(i + 1) * cells];
        // Top-down so that dp[c - w] is still the previous item's row (0/1, not unbounded).
        for c in (w..=capacity).rev() {
            let cand = dp[c - w] + u;
            if cand > dp[c] {
                dp[c] = cand;
                row[c] = 1;
            }
        }
    }
    let mut take = Vec::new();
    let mut c = capacity;
    for i in (0..n).rev() {
        if keep[i * cells + c] == 1 {
            take.push(i);
            c -= weights[i];
        }
    }
    take.reverse();
    Ok(take)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Textbook two-dimensional table with the standard backtrace, the same oracle the Python tests use.
    fn reference(weights: &[usize], utilities: &[f64], capacity: usize) -> Vec<usize> {
        let n = weights.len();
        let mut table = vec![vec![0.0f64; capacity + 1]];
        for i in 0..n {
            let prev = table[i].clone();
            let mut row = prev.clone();
            for c in weights[i]..=capacity {
                let cand = prev[c - weights[i]] + utilities[i];
                if cand > row[c] {
                    row[c] = cand;
                }
            }
            table.push(row);
        }
        let mut take = Vec::new();
        let mut c = capacity;
        for i in (0..n).rev() {
            if table[i + 1][c] != table[i][c] {
                take.push(i);
                c -= weights[i];
            }
        }
        take.reverse();
        take
    }

    /// Small deterministic LCG so the test needs no dev-dependency.
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            self.0 >> 33
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    #[test]
    fn empty_input_selects_nothing() {
        assert_eq!(exact_knapsack(&[], &[], 10).unwrap(), Vec::<usize>::new());
    }

    #[test]
    fn ties_resolve_to_earlier_items() {
        assert_eq!(
            exact_knapsack(&[5, 5, 5], &[1.0, 1.0, 1.0], 10).unwrap(),
            vec![0, 1]
        );
    }

    #[test]
    fn takes_the_better_pair_over_the_single_heavy_item() {
        assert_eq!(
            exact_knapsack(&[10, 5, 5], &[9.0, 5.0, 5.0], 10).unwrap(),
            vec![1, 2]
        );
    }

    #[test]
    fn rejects_malformed_input() {
        assert!(matches!(
            exact_knapsack(&[1, 2], &[1.0], 3),
            Err(KnapsackError::LengthMismatch { .. })
        ));
        assert!(matches!(
            exact_knapsack(&[0, 2], &[1.0, 1.0], 3),
            Err(KnapsackError::WeightOutOfRange { index: 0, .. })
        ));
        assert!(matches!(
            exact_knapsack(&[4, 2], &[1.0, 1.0], 3),
            Err(KnapsackError::WeightOutOfRange { index: 0, .. })
        ));
    }

    #[test]
    fn matches_the_textbook_table_on_random_instances() {
        let mut rng = Lcg(20261029);
        for _ in 0..300 {
            let n = 1 + rng.below(30) as usize;
            let capacity = 1 + rng.below(100) as usize;
            let weights: Vec<usize> = (0..n)
                .map(|_| 1 + rng.below(capacity as u64) as usize)
                .collect();
            let pool = [1.0, 2.0, 3.0, 5.0, 8.0];
            let utilities: Vec<f64> = (0..n)
                .map(|_| {
                    if rng.below(3) == 0 {
                        pool[rng.below(5) as usize]
                    } else {
                        rng.below(10_000) as f64 / 1000.0
                    }
                })
                .collect();
            assert_eq!(
                exact_knapsack(&weights, &utilities, capacity).unwrap(),
                reference(&weights, &utilities, capacity)
            );
        }
    }
}
