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
    call_scalar,
    call_vector,
    double_matrix,
    double_ptr,
    empty_f64,
    int32_ptr,
)
from ._ffi import check, get_lib


@dataclass(frozen=True, slots=True)
class MACDResult:
    macd: np.ndarray
    signal: np.ndarray
    histogram: np.ndarray


@dataclass(frozen=True, slots=True)
class BollingerBands:
    middle: np.ndarray
    upper: np.ndarray
    lower: np.ndarray


@dataclass(frozen=True, slots=True)
class Moments:
    mean: float
    variance: float
    stddev: float
    skewness: float
    excess_kurtosis: float
    n_valid: int


@dataclass(frozen=True, slots=True)
class TailRisk:
    var: float
    cvar: float


def _one_input(values: Any, name: str = "values") -> np.ndarray:
    return as_f64_1d(values, name)


def _series_call(
    name: str, values: Any, *arguments: Any, input_name: str = "values"
) -> np.ndarray:
    """Call a native ``(data, n, ..., out)`` kernel and return a new array."""
    data = _one_input(values, input_name)
    return call_vector(
        name, data.size, getattr(get_lib(), name),
        double_ptr(data), data.size, *arguments)


def _pair_inputs(
    x: Any, y: Any, x_name: str = "x", y_name: str = "y"
) -> tuple[np.ndarray, np.ndarray]:
    xa = _one_input(x, x_name)
    ya = _one_input(y, y_name)
    if xa.size != ya.size:
        raise ValueError(f"{x_name} and {y_name} must have equal lengths")
    return xa, ya


def _multi_output(
    name: str,
    count: int,
    size: int,
    function: Any,
    *arguments: Any,
) -> list[np.ndarray]:
    outputs = [empty_f64(size) for _ in range(count)]
    check(
        function(*arguments, *(double_ptr(output) for output in outputs)),
        name,
    )
    return outputs


def returns(prices: Any, *, log: bool = False) -> np.ndarray:
    """Simple (default) or log returns; first element is NaN."""
    return _series_call(
        "rflux_returns", prices, 1 if log else 0, input_name="prices")


def returns_panel(prices: Any, *, log: bool = False) -> np.ndarray:
    """Column-wise returns of an ``(observations, assets)`` price panel."""
    panel = as_f64_2d(prices, "prices")
    out = np.empty_like(panel, dtype=np.float64)
    for column in range(panel.shape[1]):
        out[:, column] = returns(panel[:, column], log=log)
    return out


def sma(values: Any, window: int) -> np.ndarray:
    """Simple moving average with a NaN warm-up of ``window - 1``."""
    return _series_call("rflux_sma", values, as_size(window, "window"))


def rolling_mean(values: Any, window: int) -> np.ndarray:
    """Alias for :func:`sma`."""
    return sma(values, window)


def rolling_sum(values: Any, window: int) -> np.ndarray:
    return _series_call(
        "rflux_rolling_sum", values, as_size(window, "window"))


def rolling_std(values: Any, window: int, ddof: int = 0) -> np.ndarray:
    return _series_call(
        "rflux_rolling_std", values, as_size(window, "window"),
        as_size(ddof, "ddof", minimum=0))


def rolling_min(values: Any, window: int) -> np.ndarray:
    return _series_call(
        "rflux_rolling_min", values, as_size(window, "window"))


def rolling_max(values: Any, window: int) -> np.ndarray:
    return _series_call(
        "rflux_rolling_max", values, as_size(window, "window"))


def rolling_zscore(values: Any, window: int, ddof: int = 0) -> np.ndarray:
    return _series_call(
        "rflux_rolling_zscore", values, as_size(window, "window"),
        as_size(ddof, "ddof", minimum=0))


def rolling_quantile(values: Any, window: int, q: float) -> np.ndarray:
    """Rolling Type-7 quantile over complete windows."""
    return _series_call(
        "rflux_rolling_quantile", values, as_size(window, "window"), float(q))


def rolling_median(values: Any, window: int) -> np.ndarray:
    """Rolling median, equivalent to ``rolling_quantile(..., 0.5)``."""
    return rolling_quantile(values, window, 0.5)


def ema(values: Any, span: int) -> np.ndarray:
    """EMA with alpha = 2/(span+1), seeded on the first finite value."""
    return _series_call("rflux_ema", values, as_size(span, "span"))


def rsi(prices: Any, period: int = 14) -> np.ndarray:
    return _series_call(
        "rflux_rsi", prices, as_size(period, "period"), input_name="prices")


def macd(
    prices: Any, fast: int = 12, slow: int = 26, signal: int = 9
) -> MACDResult:
    data = _one_input(prices, "prices")
    outputs = _multi_output(
        "rflux_macd",
        3,
        data.size,
        get_lib().rflux_macd,
        double_ptr(data),
        data.size,
        as_size(fast, "fast"),
        as_size(slow, "slow"),
        as_size(signal, "signal"),
    )
    return MACDResult(*outputs)


def bollinger(
    prices: Any, window: int = 20, n_std: float = 2.0
) -> BollingerBands:
    data = _one_input(prices, "prices")
    outputs = _multi_output(
        "rflux_bollinger",
        3,
        data.size,
        get_lib().rflux_bollinger,
        double_ptr(data),
        data.size,
        as_size(window, "window"),
        float(n_std),
    )
    return BollingerBands(*outputs)


def atr(
    high: Any, low: Any, close: Any, period: int = 14
) -> np.ndarray:
    high_array = _one_input(high, "high")
    low_array = _one_input(low, "low")
    close_array = _one_input(close, "close")
    if not (high_array.size == low_array.size == close_array.size):
        raise ValueError("high, low and close must have equal lengths")
    return call_vector(
        "rflux_atr", high_array.size, get_lib().rflux_atr,
        double_ptr(high_array), double_ptr(low_array), double_ptr(close_array),
        high_array.size, as_size(period, "period"))


def drawdown(equity: Any) -> np.ndarray:
    return _series_call(
        "rflux_drawdown", equity, input_name="equity")


def rolling_corr(x: Any, y: Any, window: int) -> np.ndarray:
    xa, ya = _pair_inputs(x, y)
    return call_vector(
        "rflux_rolling_corr", xa.size, get_lib().rflux_rolling_corr,
        double_ptr(xa), double_ptr(ya), xa.size, as_size(window, "window"))


def rolling_cov(
    x: Any, y: Any, window: int, ddof: int = 0
) -> np.ndarray:
    """Rolling covariance over complete windows; ``ddof`` defaults to 0."""
    xa, ya = _pair_inputs(x, y)
    return call_vector(
        "rflux_rolling_cov", xa.size, get_lib().rflux_rolling_cov,
        double_ptr(xa), double_ptr(ya), xa.size, as_size(window, "window"),
        as_size(ddof, "ddof", minimum=0))


def rolling_beta(asset: Any, benchmark: Any, window: int) -> np.ndarray:
    """Rolling OLS beta of ``asset`` on ``benchmark``; NaN for degenerate windows."""
    y, x = _pair_inputs(asset, benchmark, "asset", "benchmark")
    return call_vector(
        "rflux_rolling_beta", y.size, get_lib().rflux_rolling_beta,
        double_ptr(y), double_ptr(x), y.size, as_size(window, "window"))


def rolling_slope(values: Any, window: int) -> np.ndarray:
    return _series_call(
        "rflux_rolling_slope", values, as_size(window, "window"))


def rolling_r2(values: Any, window: int) -> np.ndarray:
    return _series_call(
        "rflux_rolling_r2", values, as_size(window, "window"))


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
    return call_scalar(
        "rflux_quantile", get_lib().rflux_quantile,
        double_ptr(data), data.size, float(q))


def pearson(x: Any, y: Any) -> float:
    xa, ya = _pair_inputs(x, y)
    return call_scalar(
        "rflux_pearson", get_lib().rflux_pearson,
        double_ptr(xa), double_ptr(ya), xa.size)


def correlation_matrix(panel: Any) -> np.ndarray:
    """Pairwise-complete correlation matrix of a nobs x nassets panel."""
    data = as_f64_2d(panel, "panel")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("panel must not be empty")
    n_assets = data.shape[1]
    out = call_vector(
        "rflux_corr_matrix", n_assets * n_assets, get_lib().rflux_corr_matrix,
        double_ptr(data), data.shape[0], n_assets)
    return double_matrix(out, n_assets)


def sharpe(
    returns_: Any, periods_per_year: float = 252.0, risk_free: float = 0.0
) -> float:
    data = _one_input(returns_, "returns")
    return call_scalar(
        "rflux_sharpe", get_lib().rflux_sharpe,
        double_ptr(data), data.size, float(periods_per_year), float(risk_free))


def sortino(
    returns_: Any, periods_per_year: float = 252.0, risk_free: float = 0.0
) -> float:
    data = _one_input(returns_, "returns")
    return call_scalar(
        "rflux_sortino", get_lib().rflux_sortino,
        double_ptr(data), data.size, float(periods_per_year), float(risk_free))


def var_cvar(returns_: Any, confidence: float = 0.95) -> TailRisk:
    """Historical VaR/CVaR; both are positive losses and may be negative."""
    data = _one_input(returns_, "returns")
    var, cvar = _multi_output(
        "rflux_var_cvar",
        2,
        1,
        get_lib().rflux_var_cvar,
        double_ptr(data),
        data.size,
        float(confidence),
    )
    return TailRisk(float(var[0]), float(cvar[0]))


def rank(values: Any) -> np.ndarray:
    """Average ascending rank; ties share their mean rank, NaN stays NaN."""
    return _series_call("rflux_rank", values)


def percentile_rank(values: Any) -> np.ndarray:
    return _series_call("rflux_percentile_rank", values)


def zscore(values: Any, ddof: int = 0) -> np.ndarray:
    return _series_call(
        "rflux_zscore", values, as_size(ddof, "ddof", minimum=0))


def winsorize(
    values: Any, lower_q: float = 0.01, upper_q: float = 0.99
) -> np.ndarray:
    return _series_call(
        "rflux_winsorize", values, float(lower_q), float(upper_q))


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
    return call_vector(
        "rflux_neutralize", data.size, get_lib().rflux_neutralize,
        double_ptr(data), data.size, int32_ptr(group_array),
        as_size(group_count, "group_count"))


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
    "returns_panel",
    "rolling_beta",
    "rolling_corr",
    "rolling_cov",
    "rolling_max",
    "rolling_mean",
    "rolling_median",
    "rolling_min",
    "rolling_quantile",
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
