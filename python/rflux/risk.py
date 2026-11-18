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

    For each pair ``(i, j)`` the lower coefficient averages the two empirical
    conditionals ``P(r_i <= Q_i(q) | r_j <= Q_j(q))`` and
    ``P(r_j <= Q_j(q) | r_i <= Q_i(q))``; the upper coefficient does the same
    for the upper tails.  Averaging makes the finite-sample matrix genuinely
    symmetric.  The diagonal is one; pairs with no observations in either
    conditioning tail are NaN.
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
            joint = int((events[:, i] & events[:, j]).sum())
            count_i = int(events[:, i].sum())
            count_j = int(events[:, j].sum())
            if count_i == 0 or count_j == 0:
                value = np.nan
            else:
                # The population coefficient is symmetric, but the two
                # finite-sample conditionals P(i | j) and P(j | i) are not.
                # Average them so the reported matrix is genuinely symmetric
                # instead of borrowing only one direction.
                value = 0.5 * (joint / count_i + joint / count_j)
            result[i, j] = value
            result[j, i] = value
    return result
