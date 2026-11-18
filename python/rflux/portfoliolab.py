"""Portfolio construction and rebalancing.

HRP, NCO, risk parity and rebalance gating are implemented in C++.  Python
wrappers use the native extension when available and the C ABI otherwise.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._common import as_f64_2d, as_finite_float, double_ptr, empty_f64
from ._ffi import c_size_p, check, get_lib
from ._native_backend import native_module, native_or

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
    matrix = _covariance(covariance)
    def fallback() -> np.ndarray:
        out = empty_f64(matrix.shape[0])
        check(
            get_lib().rflux_hrp_weights(
                double_ptr(matrix), matrix.shape[0], double_ptr(out)
            ),
            "rflux_hrp_weights",
        )
        return out

    return native_or(
        native_module,
        "rflux_hrp_weights",
        lambda: native_module.hrp_weights(matrix),
        fallback,
    )


def quasi_diagonal_order(covariance: Any) -> np.ndarray:
    """Leaf order used by the HRP recursion."""
    matrix = _covariance(covariance)
    n = matrix.shape[0]
    order = np.empty(n, dtype=np.uintp)
    check(
        get_lib().rflux_hrp_order(
            double_ptr(matrix), n, order.ctypes.data_as(c_size_p)
        ),
        "rflux_hrp_order",
    )
    return order.astype(np.int64, copy=False)


def nco_weights(
    covariance: Any, *, distance_threshold: float = 0.5
) -> np.ndarray:
    """Nested clustered risk parity, implemented in C++.

    Assets are grouped by single-linkage correlation distance, each group is
    HRP-allocated, and groups receive inverse cluster-variance budgets.
    """
    matrix = _covariance(covariance)
    threshold = as_finite_float(distance_threshold, "distance_threshold")
    if threshold < 0.0:
        raise ValueError("distance_threshold must be non-negative")
    n = matrix.shape[0]
    def fallback() -> np.ndarray:
        out = empty_f64(n)
        check(
            get_lib().rflux_nco_weights(
                double_ptr(matrix), n, float(threshold), double_ptr(out)
            ),
            "rflux_nco_weights",
        )
        return out

    return native_or(
        native_module,
        "rflux_nco_weights",
        lambda: native_module.nco_weights(matrix, float(threshold)),
        fallback,
    )


def risk_parity_weights(covariance: Any) -> np.ndarray:
    """Diagonal inverse-variance allocation, implemented in C++."""
    matrix = _covariance(covariance)
    n = matrix.shape[0]
    def fallback() -> np.ndarray:
        out = empty_f64(n)
        check(
            get_lib().rflux_risk_parity_weights(
                double_ptr(matrix), n, double_ptr(out)
            ),
            "rflux_risk_parity_weights",
        )
        return out

    return native_or(
        native_module,
        "rflux_risk_parity_weights",
        lambda: native_module.risk_parity_weights(matrix),
        fallback,
    )


def rebalance_targets(
    current: Any, target: Any, *, threshold: float = 0.05
) -> np.ndarray:
    """Gate each portfolio row independently, implemented in C++.

    A row switches from current to target when its maximum absolute drift
    reaches ``threshold``.
    """
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
    current_array = np.ascontiguousarray(current_array)
    target_array = np.ascontiguousarray(target_array)
    n_rows, n_assets = current_array.shape
    def fallback() -> np.ndarray:
        out = empty_f64(n_rows * n_assets)
        check(
            get_lib().rflux_rebalance_targets(
                double_ptr(current_array),
                double_ptr(target_array),
                n_rows,
                n_assets,
                float(threshold),
                double_ptr(out),
            ),
            "rflux_rebalance_targets",
        )
        return out.reshape(n_rows, n_assets)

    result = native_or(
        native_module,
        "rflux_rebalance_targets",
        lambda: native_module.rebalance_targets(
            current_array, target_array, float(threshold)
        ),
        fallback,
    )
    return result[0] if vector_output else result
