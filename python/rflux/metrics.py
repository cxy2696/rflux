"""Portfolio and risk metrics assembled from native qflux kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import qflux
from ._common import (
    as_f64_1d,
    as_finite_float,
    as_open_unit_float,
    as_positive_float,
    as_size,
)


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    n_periods: int
    elapsed_periods: int
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    var: float
    cvar: float
    confidence: float
    hit_rate: float


def _validated_options(
    periods_per_year: Any = 252.0,
    risk_free: Any = 0.0,
    confidence: Any | None = 0.95,
) -> tuple[float, float, float | None]:
    """Validate the annualisation/risk-free/confidence triple once."""
    annual = as_positive_float(periods_per_year, "periods_per_year")
    free = as_finite_float(risk_free, "risk_free")
    level = (
        as_open_unit_float(confidence, "confidence")
        if confidence is not None
        else None
    )
    return annual, free, level


def equity_curve(returns_: Any, initial: float = 1.0) -> np.ndarray:
    """Compounded equity curve; finite returns are compounded, NaN gaps are flat.

    Returns must be strictly greater than -1 for every finite observation;
    otherwise the compounded path is not a valid capital curve.
    """
    data = as_f64_1d(returns_, "returns")
    finite = data[np.isfinite(data)]
    if finite.size and np.any(finite <= -1.0):
        raise ValueError("returns must be greater than -1 for compounding")
    values = np.where(np.isfinite(data), 1.0 + data, 1.0)
    return float(initial) * np.cumprod(values)


def performance_summary(
    returns_: Any,
    *,
    periods_per_year: float = 252.0,
    risk_free: float = 0.0,
    confidence: float = 0.95,
    periods: int | None = None,
) -> PerformanceMetrics:
    """Annualized performance and tail-risk summary of periodic returns.

    Returns are arithmetic periodic returns (not percentages). ``risk_free`` is
    an annual rate and ``confidence`` is the historical VaR/CVaR confidence.
    NaN returns are excluded from the statistics, while ``periods`` controls the
    elapsed time used for annualising the geometric return.  By default the full
    input length is used, so internal NaN gaps do not shorten the horizon.  Pass
    ``periods=finite_count`` to annualise over observed periods instead.

    Portfolio returns may be below -100% (for example a levered long-short
    book).  In that case total_return is still reported, while annualized
    return and Calmar are NaN once cumulative equity is non-positive.
    """
    data = as_f64_1d(returns_, "returns")
    finite = data[np.isfinite(data)]
    if finite.size < 2:
        raise ValueError("at least two finite returns are required")
    periods_per_year, risk_free, confidence = _validated_options(
        periods_per_year, risk_free, confidence
    )
    assert confidence is not None
    elapsed = (
        int(data.size)
        if periods is None
        else as_size(periods, "periods")
    )
    if elapsed <= 0:
        raise ValueError("periods must be positive")

    growth = 1.0 + finite
    total_return = float(np.prod(growth) - 1.0)
    if growth.min() <= 0.0 or 1.0 + total_return <= 0.0:
        annualized_return = float("nan")
    else:
        annualized_return = float(
            (1.0 + total_return) ** (periods_per_year / elapsed) - 1.0
        )

    annualized_volatility = float(
        np.std(finite, ddof=1) * np.sqrt(periods_per_year)
    )
    equity = np.concatenate(([1.0], np.cumprod(growth)))
    drawdowns = qflux.drawdown(equity)[1:]
    finite_drawdowns = drawdowns[np.isfinite(drawdowns)]
    max_drawdown = (
        float(finite_drawdowns.min()) if finite_drawdowns.size else float("nan")
    )
    if np.isfinite(annualized_return) and max_drawdown < 0.0:
        calmar = annualized_return / abs(max_drawdown)
    else:
        calmar = float("nan")
    tail = qflux.var_cvar(finite, confidence)
    sharpe = qflux.sharpe(finite, periods_per_year, risk_free)
    sortino = qflux.sortino(finite, periods_per_year, risk_free)
    return PerformanceMetrics(
        n_periods=int(finite.size),
        elapsed_periods=int(elapsed),
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_drawdown=max_drawdown,
        calmar=float(calmar),
        var=float(tail.var),
        cvar=float(tail.cvar),
        confidence=float(confidence),
        hit_rate=float(np.mean(finite > 0.0)),
    )


def drawdown_series(equity: Any) -> np.ndarray:
    """Drawdown from the running peak."""
    return qflux.drawdown(equity)


__all__ = [
    "PerformanceMetrics",
    "drawdown_series",
    "equity_curve",
    "performance_summary",
]
