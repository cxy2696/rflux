"""qflux: vectorised factors, rolling statistics and cross-sectional tools.

Every function calls a C++ kernel through the pybind11 bridge.  Returned
arrays are new NumPy arrays; inputs are never modified.  Undefined warm-up
values and missing-data windows are ``NaN``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ._common import as_f64_1d, as_f64_2d, as_i32_1d, as_size
from ._native_backend import native_api as native_module


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
    return native_module.series(name, _one_input(values, input_name), *arguments)


def _pair_series_call(
    name: str,
    x: Any,
    y: Any,
    *arguments: Any,
    x_name: str = "x",
    y_name: str = "y",
) -> np.ndarray:
    xa = _one_input(x, x_name)
    ya = _one_input(y, y_name)
    if xa.size != ya.size:
        raise ValueError(f"{x_name} and {y_name} must have equal lengths")
    return native_module.pair_series(
        name, _one_input(x, x_name), ya, *arguments)


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
    result = native_module.macd(
        data, as_size(fast, "fast"), as_size(slow, "slow"),
        as_size(signal, "signal"),
    )
    return MACDResult(*result)


def bollinger(
    prices: Any, window: int = 20, n_std: float = 2.0
) -> BollingerBands:
    data = _one_input(prices, "prices")
    result = native_module.bollinger(data, as_size(window, "window"), float(n_std))
    return BollingerBands(*result)


def atr(
    high: Any, low: Any, close: Any, period: int = 14
) -> np.ndarray:
    high_array = _one_input(high, "high")
    low_array = _one_input(low, "low")
    close_array = _one_input(close, "close")
    if not (high_array.size == low_array.size == close_array.size):
        raise ValueError("high, low and close must have equal lengths")
    return native_module.atr(
        high_array, low_array, close_array, as_size(period, "period"))


def drawdown(equity: Any) -> np.ndarray:
    return _series_call("rflux_drawdown", equity, input_name="equity")


def rolling_corr(x: Any, y: Any, window: int) -> np.ndarray:
    return _pair_series_call(
        "rflux_rolling_corr", x, y, as_size(window, "window"))


def rolling_cov(
    x: Any, y: Any, window: int, ddof: int = 0
) -> np.ndarray:
    """Rolling covariance over complete windows; ``ddof`` defaults to 0."""
    return _pair_series_call(
        "rflux_rolling_cov", x, y, as_size(window, "window"),
        as_size(ddof, "ddof", minimum=0))


def rolling_beta(asset: Any, benchmark: Any, window: int) -> np.ndarray:
    """Rolling OLS beta of ``asset`` on ``benchmark``."""
    return _pair_series_call(
        "rflux_rolling_beta", asset, benchmark, as_size(window, "window"),
        x_name="asset", y_name="benchmark")


def rolling_slope(values: Any, window: int) -> np.ndarray:
    return _series_call(
        "rflux_rolling_slope", values, as_size(window, "window"))


def rolling_r2(values: Any, window: int) -> np.ndarray:
    return _series_call(
        "rflux_rolling_r2", values, as_size(window, "window"))


def describe(values: Any) -> Moments:
    """Finite-value descriptive statistics (sample variance, adjusted skew)."""
    data = _one_input(values)
    mean, variance, stddev, skewness, kurtosis, valid = (
        native_module.describe(data)
    )
    return Moments(
        float(mean), float(variance), float(stddev), float(skewness),
        float(kurtosis), int(valid),
    )


def quantile(values: Any, q: float) -> float:
    return float(native_module.quantile(_one_input(values), float(q)))


def pearson(x: Any, y: Any) -> float:
    xa = _one_input(x, "x")
    ya = _one_input(y, "y")
    if xa.size != ya.size:
        raise ValueError("x and y must have equal lengths")
    return float(native_module.pearson(xa, ya))


def correlation_matrix(panel: Any) -> np.ndarray:
    """Pairwise-complete correlation matrix of a nobs x nassets panel."""
    data = as_f64_2d(panel, "panel")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("panel must not be empty")
    return native_module.correlation_matrix(data)


def _performance_ratio(
    name: str, kernel: str, returns_: Any, periods_per_year: float,
    risk_free: float,
) -> float:
    data = _one_input(returns_, "returns")
    return float(getattr(native_module, kernel)(
        data, float(periods_per_year), float(risk_free)))


def sharpe(
    returns_: Any, periods_per_year: float = 252.0, risk_free: float = 0.0
) -> float:
    return _performance_ratio(
        "rflux_sharpe", "sharpe", returns_, periods_per_year, risk_free)


def sortino(
    returns_: Any, periods_per_year: float = 252.0, risk_free: float = 0.0
) -> float:
    return _performance_ratio(
        "rflux_sortino", "sortino", returns_, periods_per_year, risk_free)


def var_cvar(returns_: Any, confidence: float = 0.95) -> TailRisk:
    """Historical VaR/CVaR; both are positive losses and may be negative."""
    data = _one_input(returns_, "returns")
    var, cvar = native_module.var_cvar(data, float(confidence))
    return TailRisk(float(var), float(cvar))


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
    return native_module.neutralize(
        data, group_array, as_size(group_count, "group_count"))


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
