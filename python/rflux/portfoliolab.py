"""Portfolio construction and rebalancing.

HRP, NCO, risk parity and rebalance gating are implemented in the C++ core.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._common import as_f64_2d, as_finite_float
from ._native_backend import native_api as native_module

__all__ = [
    "hrp_weights",
    "nco_weights",
    "quasi_diagonal_order",
    "rebalance_targets",
    "risk_parity_weights",
]


def _covariance(values: Any) -> np.ndarray:
    matrix = as_f64_2d(values, "covariance")
    n = matrix.shape[0]
    if n == 0 or matrix.shape[1] != n:
        raise ValueError("covariance must be a non-empty square matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance must be finite")
    if not np.allclose(matrix, matrix.T, rtol=1e-8, atol=1e-10):
        raise ValueError("covariance must be symmetric")
    if np.any(np.diag(matrix) <= 0.0):
        raise ValueError("covariance diagonal must be positive")
    return np.ascontiguousarray(matrix)


def hrp_weights(covariance: Any) -> np.ndarray:
    """Long-only hierarchical risk parity weights summing to one."""
    return native_module.hrp_weights(_covariance(covariance))


def quasi_diagonal_order(covariance: Any) -> np.ndarray:
    """Leaf order used by the HRP recursion."""
    return np.asarray(
        native_module.hrp_order(_covariance(covariance)), dtype=np.int64
    )


def nco_weights(
    covariance: Any, *, distance_threshold: float = 0.5
) -> np.ndarray:
    """Nested clustered risk parity, implemented in C++."""
    matrix = _covariance(covariance)
    threshold = as_finite_float(distance_threshold, "distance_threshold")
    if threshold < 0.0:
        raise ValueError("distance_threshold must be non-negative")
    return native_module.nco_weights(matrix, float(threshold))


def risk_parity_weights(covariance: Any) -> np.ndarray:
    """Diagonal inverse-variance allocation, implemented in C++."""
    return native_module.risk_parity_weights(_covariance(covariance))


def rebalance_targets(
    current: Any, target: Any, *, threshold: float = 0.05
) -> np.ndarray:
    """Gate each portfolio row independently, implemented in C++."""
    current_array = np.asarray(current, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    vector_output = current_array.ndim == 1
    if vector_output:
        current_array = current_array.reshape(1, -1)
        target_array = target_array.reshape(1, -1)
    if current_array.ndim != 2 or target_array.ndim != 2:
        raise ValueError("current and target must be 1D or 2D weight arrays")
    if current_array.shape != target_array.shape:
        raise ValueError("current and target must have the same shape")
    if np.any(~np.isfinite(current_array)) or np.any(~np.isfinite(target_array)):
        raise ValueError("weights must be finite")
    threshold = as_finite_float(threshold, "threshold")
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative")
    result = native_module.rebalance_targets(
        np.ascontiguousarray(current_array),
        np.ascontiguousarray(target_array),
        float(threshold),
    )
    return result[0] if vector_output else result
