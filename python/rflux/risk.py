"""Tail-risk dependence and extreme co-movement diagnostics."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ._common import as_f64_2d, double_ptr, empty_f64
from ._ffi import check, get_lib
from ._native_backend import native_module, native_or

__all__ = ["tail_dependence"]


def tail_dependence(
    returns: Any, *, q: float = 0.05, tail: Literal["lower", "upper"] = "lower"
) -> np.ndarray:
    """Empirical pairwise tail-dependence coefficients.

    For each pair ``(i, j)`` the lower coefficient averages the two empirical
    conditionals ``P(r_i <= Q_i(q) | r_j <= Q_j(q))`` and
    ``P(r_j <= Q_j(q) | r_i <= Q_i(q))``; the upper coefficient does the same
    for the upper tails.  Averaging makes the finite-sample matrix symmetric.  The diagonal is one; pairs with no observations in either
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
    upper = tail == "upper"
    def fallback() -> np.ndarray:
        out = empty_f64(n_assets * n_assets)
        check(
            get_lib().rflux_tail_dependence(
                double_ptr(values),
                values.shape[0],
                n_assets,
                float(q),
                1 if upper else 0,
                double_ptr(out),
            ),
            "rflux_tail_dependence",
        )
        return out.reshape(n_assets, n_assets)

    return native_or(
        native_module,
        "rflux_tail_dependence",
        lambda: native_module.tail_dependence(values, float(q), upper),
        fallback,
    )
