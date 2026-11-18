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
from ._native_backend import native_api as native_module


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
    """Compounded equity curve, implemented in C++; NaN gaps are flat."""
    data = as_f64_1d(returns_, "returns")
    finite = data[np.isfinite(data)]
    if finite.size and np.any(finite <= -1.0):
        raise ValueError("returns must be greater than -1 for compounding")
    initial = as_finite_float(initial, "initial")
    return native_module.equity_curve(data, float(initial))


def performance_summary(
    returns_: Any,
    *,
    periods_per_year: float = 252.0,
    risk_free: float = 0.0,
    confidence: float = 0.95,
    periods: int | None = None,
) -> PerformanceMetrics:
    """Annualized performance and tail-risk summary, implemented in C++."""
    data = as_f64_1d(returns_, "returns")
    finite = data[np.isfinite(data)]
    if finite.size < 2:
        raise ValueError("at least two finite returns are required")
    periods_per_year, risk_free, confidence = _validated_options(
        periods_per_year, risk_free, confidence
    )
    assert confidence is not None
    elapsed = 0 if periods is None else as_size(periods, "periods")
    if periods is not None and elapsed <= 0:
        raise ValueError("periods must be positive")
    n_periods, elapsed_periods, values = native_module.performance_summary(
        data, float(periods_per_year), float(risk_free), float(confidence),
        elapsed,
    )
    return PerformanceMetrics(
        n_periods=int(n_periods),
        elapsed_periods=int(elapsed_periods),
        total_return=float(values[0]),
        annualized_return=float(values[1]),
        annualized_volatility=float(values[2]),
        sharpe=float(values[3]),
        sortino=float(values[4]),
        max_drawdown=float(values[5]),
        calmar=float(values[6]),
        var=float(values[7]),
        cvar=float(values[8]),
        confidence=float(confidence),
        hit_rate=float(values[9]),
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
