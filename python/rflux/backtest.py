"""Lightweight cross-sectional research and backtest tools.

These are transparent building blocks for research, not a live trading
engine.  Alignment is explicit: ``factor[t]`` must be known before the return
realized over ``forward_returns[t]``.  No future information is read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import qflux
from ._common import as_f64_1d, as_f64_2d, as_positive_float, as_size
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
    out = np.full_like(array, np.nan, dtype=np.float64)
    if array.shape[0] <= periods:
        return out
    future = array[periods:]
    past = array[:-periods]
    base_valid = np.isfinite(future) & np.isfinite(past)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = future / past
        if log:
            valid = base_valid & (future > 0) & (past > 0)
            out[:-periods] = np.where(valid, np.log(ratio), np.nan)
        else:
            valid = base_valid & (past != 0)
            out[:-periods] = np.where(valid, ratio - 1.0, np.nan)
    return out


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


def _buckets(
    factor_row: np.ndarray, forward_row: np.ndarray, n_quantiles: int
) -> list[np.ndarray]:
    valid = np.isfinite(factor_row) & np.isfinite(forward_row)
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        return [np.empty(0, dtype=np.int64) for _ in range(n_quantiles)]
    order = np.argsort(factor_row[indices], kind="mergesort")
    sorted_indices = indices[order]
    return [
        part.astype(np.int64)
        for part in np.array_split(sorted_indices, n_quantiles)
    ]


def _bucket_means(
    forward_row: np.ndarray, buckets: list[np.ndarray]
) -> np.ndarray:
    means = np.full(len(buckets), np.nan, dtype=np.float64)
    for index, members in enumerate(buckets):
        if members.size:
            means[index] = float(np.mean(forward_row[members]))
    return means


def quantile_portfolio_returns(
    factor: Any, forward_returns_: Any, n_quantiles: int = 5
) -> np.ndarray:
    """Equal-weight forward returns by factor quantile.

    Returns shape ``(n_quantiles, n_periods)``; row 0 is the low-factor bucket
    and the last row is the high-factor bucket.  Empty buckets are NaN.
    """
    q = as_size(n_quantiles, "n_quantiles", minimum=2)
    factor_array, forward_array = _validate_panels(factor, forward_returns_)
    out = np.full((q, factor_array.shape[0]), np.nan)
    for t in range(factor_array.shape[0]):
        buckets = _buckets(factor_array[t], forward_array[t], q)
        out[:, t] = _bucket_means(forward_array[t], buckets)
    return out


def factor_ic(
    factor: Any, forward_returns_: Any, *, method: str = "spearman"
) -> np.ndarray:
    """Cross-sectional information coefficient for each period.

    ``spearman`` ranks both sides first; ``pearson`` uses raw values.  Periods
    with fewer than three valid pairs are NaN.
    """
    factor_array, forward_array = _validate_panels(factor, forward_returns_)
    normalized = str(method).lower()
    if normalized not in {"spearman", "pearson"}:
        raise ValueError("method must be 'spearman' or 'pearson'")
    ic = np.full(factor_array.shape[0], np.nan)
    for t in range(factor_array.shape[0]):
        valid = np.isfinite(factor_array[t]) & np.isfinite(forward_array[t])
        if np.count_nonzero(valid) < 3:
            continue
        if normalized == "spearman":
            x = qflux.rank(factor_array[t, valid])
            y = qflux.rank(forward_array[t, valid])
        else:
            x = factor_array[t, valid]
            y = forward_array[t, valid]
        value = qflux.pearson(x, y)
        if np.isfinite(value):
            ic[t] = value
    return ic


def summarize_ic(ic: Any) -> ICSummary:
    """Summarize IC values without annualising the mean/std ratio."""
    values = as_f64_1d(ic, "ic")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ICSummary(float("nan"), float("nan"), float("nan"), float("nan"), 0)
    std = float(np.std(finite, ddof=1)) if finite.size >= 2 else float("nan")
    mean = float(np.mean(finite))
    information_ratio = (
        mean / std if std and std > 0 and np.isfinite(std) else float("nan")
    )
    return ICSummary(
        mean=mean,
        std=std,
        information_ratio=float(information_ratio),
        positive_ratio=float(np.mean(finite > 0.0)),
        n_periods=int(finite.size),
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

    Portfolio construction for each period:

    - eligible assets need finite factor and finite forward return;
    - assets are sorted by factor and split into ``n_quantiles`` groups;
    - the highest non-empty group is held long at +1 gross and the lowest at
      -1 gross (each leg equal weighted);
    - one-way turnover is ``0.5 * sum(abs(w_t - w_{t-1}))`` over both legs;
    - ``cost_bps`` is charged on that turnover.

    Periods without at least two non-empty groups are flat (gross return 0), but
    closing a previous position still incurs cost.  This keeps the capital path
    defined for sparse cross-sections.  A long-short portfolio can lose more
    than 100% of its initial capital; the equity path is still returned, while
    geometric return/Calmar become NaN once equity is non-positive.
    """
    q = as_size(n_quantiles, "n_quantiles", minimum=2)
    if not np.isfinite(cost_bps) or cost_bps < 0.0:
        raise ValueError("cost_bps must be finite and non-negative")
    periods_per_year = as_positive_float(periods_per_year, "periods_per_year")
    factor_array, forward_array = _validate_panels(factor, forward_returns_)
    periods, assets = factor_array.shape

    gross = np.zeros(periods, dtype=np.float64)
    costs = np.zeros(periods, dtype=np.float64)
    turnover = np.zeros(periods, dtype=np.float64)
    quantile_returns = np.full((q, periods), np.nan, dtype=np.float64)
    previous_weights = np.zeros(assets, dtype=np.float64)

    for t in range(periods):
        buckets = _buckets(factor_array[t], forward_array[t], q)
        non_empty = [bucket for bucket in buckets if bucket.size]
        quantile_returns[:, t] = _bucket_means(forward_array[t], buckets)

        weights = np.zeros(assets, dtype=np.float64)
        if len(non_empty) >= 2:
            low = non_empty[0]
            high = non_empty[-1]
            weights[high] = 1.0 / high.size
            weights[low] = -1.0 / low.size
            gross[t] = float(
                np.mean(forward_array[t, high])
                - np.mean(forward_array[t, low])
            )
        turnover[t] = 0.5 * float(np.sum(np.abs(weights - previous_weights)))
        costs[t] = turnover[t] * float(cost_bps) / 10000.0
        previous_weights = weights

    net = gross - costs
    equity = np.empty(periods + 1, dtype=np.float64)
    equity[0] = 1.0
    equity[1:] = np.cumprod(1.0 + net)
    metrics = (
        performance_summary(net, periods_per_year=periods_per_year)
        if periods >= 2
        else None
    )
    return BacktestResult(
        gross_returns=gross,
        costs=costs,
        net_returns=net,
        turnover=turnover,
        equity=equity,
        quantile_returns=quantile_returns,
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
