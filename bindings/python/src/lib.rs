//! Python extension over the `harpe` crate's kernels (issue #11).

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Exact 0/1 knapsack: indices (ascending) of the maximum-utility subset whose weights sum to at
/// most `capacity`. Every weight must lie in `1..=capacity`; ties resolve to earlier items, as in
/// the Python kernel. Raises `ValueError` on malformed input.
#[pyfunction]
fn exact_knapsack(weights: Vec<usize>, utilities: Vec<f64>, capacity: usize) -> PyResult<Vec<usize>> {
    harpe::exact_knapsack(&weights, &utilities, capacity).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Version of the Rust crate this extension was built from.
#[pyfunction]
fn crate_version() -> &'static str {
    harpe::VERSION
}

#[pymodule]
fn harpe_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(exact_knapsack, m)?)?;
    m.add_function(wrap_pyfunction!(crate_version, m)?)?;
    Ok(())
}
