"""qflux: vectorised factors, rolling statistics and cross-sectional tools.

All functions call the C++ kernels through the stable C ABI.  Returned arrays
are new NumPy arrays; inputs are never modified.  Undefined warm-up values and
missing-data windows are ``NaN``, following the native library convention.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np

from ._common import (
    as_f64_1d,
    as_f64_2d,
    as_i32_1d,
    as_size,
    double_matrix,
    double_ptr,
    empty_f64,
    int32_ptr,
)
from ._ffi import check, get_lib


@dataclass(frozen=True)
class MACDResult:
    macd: np.ndarray
    signal: np.ndarray
    histogram: np.ndarray


@dataclass(frozen=True)
class BollingerBands:
    middle: np.ndarray
    upper: np.ndarray
    lower: np.ndarray


@dataclass(frozen=True)
class Moments:
    mean: float
    variance: float
    stddev: float
    skewness: float
    excess_kurtosis: float
    n_valid: int


@dataclass(frozen=True)
class TailRisk:
    var: float
    cvar: float


def _one_input(values: Any, name: str = "values") -> np.ndarray:
    return as_f64_1d(values, name)


def returns(prices: Any, *, log: bool = False) -> np.ndarray:
    """Simple (default) or log returns; first element is NaN."""
    values = _one_input(prices, "prices")
    out = empty_f64(values.size)
    check(
        get_lib().rflux_returns(
            double_ptr(values), values.size, 1 if log else 0, double_ptr(out)
        ),
        "rflux_returns",
    )
    return out


def sma(values: Any, window: int) -> np.ndarray:
    """Simple moving average with a NaN warm-up of ``window - 1``."""
    data = _one_input(values)
    out = empty_f64(data.size)
    check(
        get_lib().rflux_sma(double_ptr(data), data.size, as_size(window, "window"), double_ptr(out)),
        "rflux_sma",
    )
    return out


def rolling_mean(values: Any, window: int) -> np.ndarray:
    """Alias for :func:`sma`."""
    return sma(values, window)


def rolling_sum(values: Any, window: int) -> np.ndarray:
    data = _one_input(values)
    out = empty_f64(data.size)
    check(
        get_lib().rflux_rolling_sum(
            double_ptr(data), data.size, as_size(window, "window"), double_ptr(out)
        ),
        "rflux_rolling_sum",
    )
    return out


def rolling_std(values: Any, window: int, ddof: int = 0) -> np.ndarray:
    data = _one_input(values)
    out = empty_f64(data.size)
    check(
        get_lib().rflux_rolling_std(
            double_ptr(data), data.size, as_size(window, "window"), as_size(ddof, "ddof", minimum=0), double_ptr(out)
        ),
        "rflux_rolling_std",
    )
    return out


def rolling_min(values: Any, window: int) -> np.ndarray:
    data = _one_input(values)
    out = empty_f64(data.size)
    check(
        get_lib().rflux_rolling_min(
            double_ptr(data), data.size, as_size(window, "window"), double_ptr(out)
        ),
        "rflux_rolling_min",
    )
    return out


def rolling_max(values: Any, window: int) -> np.ndarray:
    data = _one_input(values)
    out = empty_f64(data.size)
    check(
        get_lib().rflux_rolling_max(
            double_ptr(data), data.size, as_size(window, "window"), double_ptr(out)
        ),
        "rflux_rolling_max",
    )
    return out


def rolling_zscore(values: Any, window: int, ddof: int = 0) -> np.ndarray:
    data = _one_input(values)
    out = empty_f64(data.size)
    check(
        get_lib().rflux_rolling_zscore(
            double_ptr(data), data.size, as_size(window, "window"), as_size(ddof, "ddof", minimum=0), double_ptr(out)
        ),
        "rflux_rolling_zscore",
    )
    return out


def ema(values: Any, span: int) -> np.ndarray:
    """EMA with alpha = 2/(span+1), seeded on the first finite value."""
    data = _one_input(values)
    out = empty_f64(data.size)
    check(
        get_lib().rflux_ema(double_ptr(data), data.size, as_size(span, "span"), double_ptr(out)),
        "rflux_ema",
    )
    return out


def rsi(prices: Any, period: int = 14) -> np.ndarray:
    data = _one_input(prices, "prices")
    out = empty_f64(data.size)
    check(
        get_lib().rflux_rsi(double_ptr(data), data.size, as_size(period, "period"), double_ptr(out)),
        "rflux_rsi",
    )
    return out


def macd(
    prices: Any, fast: int = 12, slow: int = 26, signal: int = 9
) -> MACDResult:
    data = _one_input(prices, "prices")
    n = data.size
    macd_line = empty_f64(n)
    signal_line = empty_f64(n)
    histogram = empty_f64(n)
    check(
        get_lib().rflux_macd(
            double_ptr(data),
            n,
            as_size(fast, "fast"),
            as_size(slow, "slow"),
            as_size(signal, "signal"),
            double_ptr(macd_line),
            double_ptr(signal_line),
            double_ptr(histogram),
        ),
        "rflux_macd",
    )
    return MACDResult(macd_line, signal_line, histogram)


def bollinger(
    prices: Any, window: int = 20, n_std: float = 2.0
) -> BollingerBands:
    data = _one_input(prices, "prices")
    n = data.size
    middle = empty_f64(n)
    upper = empty_f64(n)
    lower = empty_f64(n)
    check(
        get_lib().rflux_bollinger(
            double_ptr(data),
            n,
            as_size(window, "window"),
            float(n_std),
            double_ptr(middle),
            double_ptr(upper),
            double_ptr(lower),
        ),
        "rflux_bollinger",
    )
    return BollingerBands(middle, upper, lower)


def atr(
    high: Any, low: Any, close: Any, period: int = 14
) -> np.ndarray:
    high_array = _one_input(high, "high")
    low_array = _one_input(low, "low")
    close_array = _one_input(close, "close")
    if not (high_array.size == low_array.size == close_array.size):
        raise ValueError("high, low and close must have equal lengths")
    out = empty_f64(high_array.size)
    check(
        get_lib().rflux_atr(
            double_ptr(high_array),
            double_ptr(low_array),
            double_ptr(close_array),
            high_array.size,
            as_size(period, "period"),
            double_ptr(out),
        ),
        "rflux_atr",
    )
    return out


def drawdown(equity: Any) -> np.ndarray:
    data = _one_input(equity, "equity")
    out = empty_f64(data.size)
    check(
        get_lib().rflux_drawdown(
            double_ptr(data), data.size, double_ptr(out)
        ),
        "rflux_drawdown",
    )
    return out


def rolling_corr(x: Any, y: Any, window: int) -> np.ndarray:
    xa = _one_input(x, "x")
    ya = _one_input(y, "y")
    if xa.size != ya.size:
        raise ValueError("x and y must have equal lengths")
    out = empty_f64(xa.size)
    check(
        get_lib().rflux_rolling_corr(
            double_ptr(xa), double_ptr(ya), xa.size, as_size(window, "window"), double_ptr(out)
        ),
        "rflux_rolling_corr",
    )
    return out


def rolling_beta(asset: Any, benchmark: Any, window: int) -> np.ndarray:
    """Rolling OLS beta of ``asset`` on ``benchmark``; NaN for degenerate windows."""
    y = _one_input(asset, "asset")
    x = _one_input(benchmark, "benchmark")
    if y.size != x.size:
        raise ValueError("asset and benchmark must have equal lengths")
    out = empty_f64(y.size)
    check(
        get_lib().rflux_rolling_beta(
            double_ptr(y),
            double_ptr(x),
            y.size,
            as_size(window, "window"),
            double_ptr(out),
        ),
        "rflux_rolling_beta",
    )
    return out


def rolling_slope(values: Any, window: int) -> np.ndarray:
    data = _one_input(values)
    out = empty_f64(data.size)
    check(
        get_lib().rflux_rolling_slope(
            double_ptr(data), data.size, as_size(window, "window"), double_ptr(out)
        ),
        "rflux_rolling_slope",
    )
    return out


def rolling_r2(values: Any, window: int) -> np.ndarray:
    data = _one_input(values)
    out = empty_f64(data.size)
    check(
        get_lib().rflux_rolling_r2(
            double_ptr(data), data.size, as_size(window, "window"), double_ptr(out)
        ),
        "rflux_rolling_r2",
    )
    return out


def describe(values: Any) -> Moments:
    """Finite-value descriptive statistics (sample variance, adjusted skew)."""
    data = _one_input(values)
    mean = np.empty(1, dtype=np.float64)
    variance = np.empty(1, dtype=np.float64)
    stddev = np.empty(1, dtype=np.float64)
    skewness = np.empty(1, dtype=np.float64)
    kurtosis = np.empty(1, dtype=np.float64)
    valid = ctypes.c_size_t(0)
    check(
        get_lib().rflux_describe(
            double_ptr(data),
            data.size,
            double_ptr(mean),
            double_ptr(variance),
            double_ptr(stddev),
            double_ptr(skewness),
            double_ptr(kurtosis),
            ctypes.byref(valid),
        ),
        "rflux_describe",
    )
    return Moments(
        float(mean[0]),
        float(variance[0]),
        float(stddev[0]),
        float(skewness[0]),
        float(kurtosis[0]),
        int(valid.value),
    )


def quantile(values: Any, q: float) -> float:
    data = _one_input(values)
    out = np.empty(1, dtype=np.float64)
    check(
        get_lib().rflux_quantile(
            double_ptr(data), data.size, float(q), double_ptr(out)
        ),
        "rflux_quantile",
    )
    return float(out[0])


def pearson(x: Any, y: Any) -> float:
    xa = _one_input(x, "x")
    ya = _one_input(y, "y")
    if xa.size != ya.size:
        raise ValueError("x and y must have equal lengths")
    out = np.empty(1, dtype=np.float64)
    check(
        get_lib().rflux_pearson(
            double_ptr(xa), double_ptr(ya), xa.size, double_ptr(out)
        ),
        "rflux_pearson",
    )
    return float(out[0])


def correlation_matrix(panel: Any) -> np.ndarray:
    """Pairwise-complete correlation matrix of a nobs x nassets panel."""
    data = as_f64_2d(panel, "panel")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("panel must not be empty")
    n_assets = data.shape[1]
    out = empty_f64(n_assets * n_assets)
    check(
        get_lib().rflux_corr_matrix(
            double_ptr(data),
            data.shape[0],
            n_assets,
            double_ptr(out),
        ),
        "rflux_corr_matrix",
    )
    return double_matrix(out, n_assets)


def sharpe(
    returns_: Any, periods_per_year: float = 252.0, risk_free: float = 0.0
) -> float:
    data = _one_input(returns_, "returns")
    out = np.empty(1, dtype=np.float64)
    check(
        get_lib().rflux_sharpe(
            double_ptr(data),
            data.size,
            float(periods_per_year),
            float(risk_free),
            double_ptr(out),
        ),
        "rflux_sharpe",
    )
    return float(out[0])


def sortino(
    returns_: Any, periods_per_year: float = 252.0, risk_free: float = 0.0
) -> float:
    data = _one_input(returns_, "returns")
    out = np.empty(1, dtype=np.float64)
    check(
        get_lib().rflux_sortino(
            double_ptr(data),
            data.size,
            float(periods_per_year),
            float(risk_free),
            double_ptr(out),
        ),
        "rflux_sortino",
    )
    return float(out[0])


def var_cvar(returns_: Any, confidence: float = 0.95) -> TailRisk:
    """Historical VaR/CVaR; both are positive losses and may be negative."""
    data = _one_input(returns_, "returns")
    var = np.empty(1, dtype=np.float64)
    cvar = np.empty(1, dtype=np.float64)
    check(
        get_lib().rflux_var_cvar(
            double_ptr(data), data.size, float(confidence), double_ptr(var),
            double_ptr(cvar)
        ),
        "rflux_var_cvar",
    )
    return TailRisk(float(var[0]), float(cvar[0]))


def rank(values: Any) -> np.ndarray:
    """Average ascending rank; ties share their mean rank, NaN stays NaN."""
    data = _one_input(values)
    out = empty_f64(data.size)
    check(
        get_lib().rflux_rank(double_ptr(data), data.size, double_ptr(out)),
        "rflux_rank",
    )
    return out


def percentile_rank(values: Any) -> np.ndarray:
    data = _one_input(values)
    out = empty_f64(data.size)
    check(
        get_lib().rflux_percentile_rank(
            double_ptr(data), data.size, double_ptr(out)
        ),
        "rflux_percentile_rank",
    )
    return out


def zscore(values: Any, ddof: int = 0) -> np.ndarray:
    data = _one_input(values)
    out = empty_f64(data.size)
    check(
        get_lib().rflux_zscore(
            double_ptr(data), data.size, as_size(ddof, "ddof", minimum=0), double_ptr(out)
        ),
        "rflux_zscore",
    )
    return out


def winsorize(values: Any, lower_q: float = 0.01, upper_q: float = 0.99) -> np.ndarray:
    data = _one_input(values)
    out = empty_f64(data.size)
    check(
        get_lib().rflux_winsorize(
            double_ptr(data),
            data.size,
            float(lower_q),
            float(upper_q),
            double_ptr(out),
        ),
        "rflux_winsorize",
    )
    return out


def neutralize(
    values: Any, groups: Any, group_count: int | None = None
) -> np.ndarray:
    """Demean within groups; use group label -1 to exclude an asset."""
    data = _one_input(values)
    group_array = as_i32_1d(groups, "groups")
    if data.size != group_array.size:
        raise ValueError("values and groups must have equal lengths")
    if group_count is None:
        valid = group_array[group_array >= 0]
        group_count = int(valid.max()) + 1 if valid.size else 0
    out = empty_f64(data.size)
    check(
        get_lib().rflux_neutralize(
            double_ptr(data),
            data.size,
            int32_ptr(group_array),
            as_size(group_count, "group_count"),
            double_ptr(out),
        ),
        "rflux_neutralize",
    )
    return out


__all__ = [
    "BollingerBands",
    "MACDResult",
    "Moments",
    "TailRisk",
    "atr",
    "bollinger",
    "correlation_matrix",
    "describe",
    "drawdown",
    "ema",
    "macd",
    "neutralize",
    "pearson",
    "percentile_rank",
    "quantile",
    "rank",
    "returns",
    "rolling_beta",
    "rolling_corr",
    "rolling_max",
    "rolling_mean",
    "rolling_min",
    "rolling_r2",
    "rolling_slope",
    "rolling_std",
    "rolling_sum",
    "rolling_zscore",
    "rsi",
    "sharpe",
    "sma",
    "sortino",
    "var_cvar",
    "winsorize",
    "zscore",
]
