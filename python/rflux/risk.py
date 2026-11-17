"""Tail-risk dependence and extreme co-movement diagnostics."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ._common import as_f64_2d

__all__ = ["tail_dependence"]


def tail_dependence(
    returns: Any, *, q: float = 0.05, tail: Literal["lower", "upper"] = "lower"
) -> np.ndarray:
    """Empirical pairwise tail-dependence coefficients.

    For each pair ``(i, j)`` the lower coefficient is
    ``P(r_i <= Q_i(q) | r_j <= Q_j(q))`` with empirical quantiles.  The upper
    coefficient uses the corresponding upper tails.  The matrix is symmetric
    with a unit diagonal; pairs with no observations in the conditioning tail
    are NaN.
    """
    values = as_f64_2d(returns, "returns")
    if values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("returns must have at least two rows and one asset")
    q = float(q)
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    tail = str(tail).lower()
    if tail not in {"lower", "upper"}:
        raise ValueError("tail must be 'lower' or 'upper'")
    n_assets = values.shape[1]
    result = np.eye(n_assets, dtype=np.float64)
    threshold = q if tail == "lower" else 1.0 - q
    events = np.empty(values.shape, dtype=bool)
    for asset in range(n_assets):
        column = values[:, asset]
        valid = np.isfinite(column)
        if not np.any(valid):
            events[:, asset] = False
            continue
        cutoff = float(np.quantile(column[valid], threshold))
        events[:, asset] = valid & (
            column <= cutoff if tail == "lower" else column >= cutoff
        )
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            denominator = int(events[:, j].sum())
            if denominator == 0:
                value = np.nan
            else:
                value = float((events[:, i] & events[:, j]).sum()) / denominator
            result[i, j] = value
            result[j, i] = value
    return result
