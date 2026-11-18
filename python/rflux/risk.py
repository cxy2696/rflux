"""Tail-risk dependence and extreme co-movement diagnostics."""

from __future__ import annotations

from typing import Any, Literal

from ._common import as_f64_2d
from ._native_backend import native_api as native_module

__all__ = ["tail_dependence"]


def tail_dependence(
    returns: Any, *, q: float = 0.05, tail: Literal["lower", "upper"] = "lower"
) -> Any:
    """Empirical pairwise tail-dependence coefficients.

    For each pair the coefficient averages the two empirical conditionals;
    the result is symmetric with a unit diagonal.  Pairs without observations
    in either conditioning tail are NaN.
    """
    values = as_f64_2d(returns, "returns")
    if values.shape[0] < 2 or values.shape[1] == 0:
        raise ValueError("returns must have at least two rows and one asset")
    q = float(q)
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    normalized = str(tail).lower()
    if normalized not in {"lower", "upper"}:
        raise ValueError("tail must be 'lower' or 'upper'")
    return native_module.tail_dependence(values, q, normalized == "upper")
