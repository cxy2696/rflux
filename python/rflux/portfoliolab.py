"""Portfolio construction and rebalancing.

``hrp`` is implemented in C++ and uses a correlation-distance single-linkage
tree plus recursive bisection.  The C ABI is used when the native extension is
not available.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._common import as_f64_2d, as_finite_float, as_size, double_ptr, empty_f64
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
    """Nested clustered risk parity (NCO-style, long-only).

    Assets are grouped by single-linkage correlation distance, each cluster is
    allocated with HRP, and clusters are risk-budgeted by inverse cluster
    variance.  This is a deterministic nested-risk-parity approximation; a full
    NCO implementation replaces the cluster stage with a constrained convex
    optimiser.
    """
    matrix = _covariance(covariance)
    threshold = as_finite_float(distance_threshold, "distance_threshold")
    if threshold < 0.0:
        raise ValueError("distance_threshold must be non-negative")
    n = matrix.shape[0]
    diagonal = np.sqrt(np.diag(matrix))
    correlation = matrix / np.outer(diagonal, diagonal)
    correlation = np.clip(correlation, -1.0, 1.0)
    distance = np.sqrt(np.maximum(0.0, 0.5 * (1.0 - correlation)))

    parent = list(range(n))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a: int, b: int) -> None:
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            parent[max(root_a, root_b)] = min(root_a, root_b)

    for i in range(n):
        for j in range(i + 1, n):
            if distance[i, j] <= threshold:
                union(i, j)
    groups: dict[int, list[int]] = {}
    for index in range(n):
        groups.setdefault(find(index), []).append(index)
    result = np.zeros(n, dtype=np.float64)
    cluster_variances: list[float] = []
    cluster_members = list(groups.values())
    for members in cluster_members:
        if len(members) == 1:
            intra = np.ones(1, dtype=np.float64)
        else:
            intra = hrp_weights(matrix[np.ix_(members, members)])
        sub = matrix[np.ix_(members, members)]
        cluster_variances.append(float(intra @ sub @ intra))
    inverse = np.asarray(
        [1.0 / value if np.isfinite(value) and value > 0.0 else 0.0
         for value in cluster_variances],
        dtype=np.float64,
    )
    if float(inverse.sum()) <= 0.0:
        inverse = np.ones_like(inverse)
    cluster_budgets = inverse / float(inverse.sum())
    for members, budget in zip(cluster_members, cluster_budgets):
        if len(members) == 1:
            result[members[0]] = budget
            continue
        sub = matrix[np.ix_(members, members)]
        intra = hrp_weights(sub)
        result[members] = budget * intra
    return result


def risk_parity_weights(covariance: Any) -> np.ndarray:
    """Diagonal inverse-variance allocation, a simple risk-parity baseline."""
    matrix = _covariance(covariance)
    inverse = 1.0 / np.diag(matrix)
    return inverse / float(inverse.sum())


def rebalance_targets(
    current: Any, target: Any, *, threshold: float = 0.05
) -> np.ndarray:
    """Return target weights only when their row's maximum drift clears ``threshold``.

    For a 2D path, each portfolio row is gated independently.
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
    # A 2D input is a path of portfolios: gate every row on its own maximum
    # drift.  Using one global maximum would force all periods to trade when a
    # single period breaches the threshold.
    drift = np.max(np.abs(target_array - current_array), axis=1, keepdims=True)
    result = np.where(drift >= threshold, target_array, current_array)
    return result[0] if vector_output else result
