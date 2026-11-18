"""Lightweight cross-sectional research and backtest tools.

Alignment is explicit: ``factor[t]`` must be known before the return realized
over ``forward_returns[t]``.  The return, bucket, IC-summary and long-short
kernels live in the C++ core (``src/factorlab/backtest.cpp``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import factorlab
from ._common import (
    as_f64_1d,
    as_f64_2d,
    as_finite_float,
    as_positive_float,
    as_size,
)
from ._native_backend import native_api as native_module
from .metrics import PerformanceMetrics, performance_summary


@dataclass(frozen=True, slots=True)
class BacktestResult:
    gross_returns: np.ndarray
    costs: np.ndarray
    net_returns: np.ndarray
    turnover: np.ndarray
    equity: np.ndarray
    quantile_returns: np.ndarray
    metrics: PerformanceMetrics | None


@dataclass(frozen=True, slots=True)
class ICSummary:
    """Per-period IC summary; ``information_ratio`` is mean/std, not annualised."""

    mean: float
    std: float
    information_ratio: float
    positive_ratio: float
    n_periods: int


def forward_returns(
    prices: Any, periods: int = 1, *, log: bool = False
) -> np.ndarray:
    """Forward simple/log returns over ``periods`` bars; the tail is NaN."""
    array = np.asarray(prices, dtype=np.float64)
    if array.ndim not in (1, 2):
        raise ValueError("prices must be 1D or 2D")
    periods = as_size(periods, "periods")
    if array.size == 0:
        return np.full(array.shape, np.nan, dtype=np.float64)
    rows = array.shape[0]
    columns = 1 if array.ndim == 1 else array.shape[1]
    panel = np.ascontiguousarray(array.reshape(rows, columns))
    values = native_module.factor_forward_returns(panel, periods, log)
    return np.asarray(values, dtype=np.float64).reshape(array.shape)


def _validate_panels(factor: Any, forward: Any) -> tuple[np.ndarray, np.ndarray]:
    factor_array = as_f64_2d(factor, "factor")
    forward_array = as_f64_2d(forward, "forward_returns")
    if factor_array.shape != forward_array.shape:
        raise ValueError(
            "factor and forward_returns must have the same shape, got "
            f"{factor_array.shape} and {forward_array.shape}"
        )
    if factor_array.shape[0] == 0 or factor_array.shape[1] == 0:
        raise ValueError("panels must not be empty")
    return factor_array, forward_array


def quantile_portfolio_returns(
    factor: Any, forward_returns_: Any, n_quantiles: int = 5
) -> np.ndarray:
    """Equal-weight forward returns by factor quantile.

    Returns shape ``(n_quantiles, n_periods)``; row 0 is the low-factor bucket
    and the last row is the high-factor bucket.  Empty buckets are NaN.
    """
    q = as_size(n_quantiles, "n_quantiles", minimum=2)
    factor_array, forward_array = _validate_panels(factor, forward_returns_)
    values = native_module.factor_quantile_returns(
        factor_array, forward_array, q
    )
    return np.asarray(values, dtype=np.float64).reshape(
        q, factor_array.shape[0]
    )


def factor_ic(
    factor: Any, forward_returns_: Any, *, method: str = "spearman"
) -> np.ndarray:
    """Cross-sectional information coefficient per period, computed in C++."""
    factor_array, forward_array = _validate_panels(factor, forward_returns_)
    normalized = str(method).lower()
    if normalized not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")
    return factorlab.information_coefficient(
        factor_array, forward_array, method=normalized, min_pairs=3
    )


def summarize_ic(ic: Any) -> ICSummary:
    """Summarize IC values without annualising the mean/std ratio."""
    values = as_f64_1d(ic, "ic")
    mean, std, information_ratio, positive_ratio, periods = (
        native_module.ic_summary(values)
    )
    return ICSummary(
        mean=float(mean),
        std=float(std),
        information_ratio=float(information_ratio),
        positive_ratio=float(positive_ratio),
        n_periods=int(periods),
    )


def long_short_backtest(
    factor: Any,
    forward_returns_: Any,
    *,
    n_quantiles: int = 5,
    cost_bps: float = 0.0,
    periods_per_year: float = 252.0,
) -> BacktestResult:
    """Equal-weight top-minus-bottom factor portfolio.

    Periods without at least two non-empty groups are flat, but closing a
    previous position still incurs cost.  A long-short portfolio can lose more
    than 100% of its initial capital; the equity path is still returned while
    geometric return/Calmar become NaN once equity is non-positive.
    """
    q = as_size(n_quantiles, "n_quantiles", minimum=2)
    cost_bps = as_finite_float(cost_bps, "cost_bps")
    if cost_bps < 0.0:
        raise ValueError("cost_bps must be finite and non-negative")
    periods_per_year = as_positive_float(periods_per_year, "periods_per_year")
    factor_array, forward_array = _validate_panels(factor, forward_returns_)
    periods = factor_array.shape[0]
    gross, costs, turnover, net, equity, quantiles = (
        native_module.factor_long_short(
            factor_array, forward_array, q, cost_bps
        )
    )
    net = np.asarray(net, dtype=np.float64)
    metrics = (
        performance_summary(net, periods_per_year=periods_per_year)
        if periods >= 2
        else None
    )
    return BacktestResult(
        gross_returns=np.asarray(gross, dtype=np.float64),
        costs=np.asarray(costs, dtype=np.float64),
        net_returns=net,
        turnover=np.asarray(turnover, dtype=np.float64),
        equity=np.asarray(equity, dtype=np.float64),
        quantile_returns=np.asarray(quantiles, dtype=np.float64).reshape(
            q, periods
        ),
        metrics=metrics,
    )


__all__ = [
    "BacktestResult",
    "ICSummary",
    "factor_ic",
    "forward_returns",
    "long_short_backtest",
    "quantile_portfolio_returns",
    "summarize_ic",
]
