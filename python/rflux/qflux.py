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
from ._native_backend import native_module, native_or


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
    """Call a native or C-ABI ``(data, n, ..., out)`` kernel."""
    data = _one_input(values, input_name)
    return native_or(
        native_module,
        name,
        lambda: native_module.series(name, data, *arguments),
        lambda: call_vector(
            name, data.size, getattr(get_lib(), name),
            double_ptr(data), data.size, *arguments),
    )


def _pair_inputs(
    x: Any, y: Any, x_name: str = "x", y_name: str = "y"
) -> tuple[np.ndarray, np.ndarray]:
    xa = _one_input(x, x_name)
    ya = _one_input(y, y_name)
    if xa.size != ya.size:
        raise ValueError(f"{x_name} and {y_name} must have equal lengths")
    return xa, ya


def _pair_series_call(
    name: str,
    x: Any,
    y: Any,
    *arguments: Any,
    x_name: str = "x",
    y_name: str = "y",
) -> np.ndarray:
    """Call a native two-input series kernel."""
    xa, ya = _pair_inputs(x, y, x_name, y_name)
    return native_or(
        native_module,
        name,
        lambda: native_module.pair_series(name, xa, ya, *arguments),
        lambda: call_vector(
            name, xa.size, getattr(get_lib(), name),
            double_ptr(xa), double_ptr(ya), xa.size, *arguments),
    )


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
    fast = as_size(fast, "fast")
    slow = as_size(slow, "slow")
    signal = as_size(signal, "signal")
    return native_or(
        native_module,
        "rflux_macd",
        lambda: MACDResult(*native_module.macd(data, fast, slow, signal)),
        lambda: MACDResult(*_multi_output(
            "rflux_macd", 3, data.size, get_lib().rflux_macd,
            double_ptr(data), data.size, fast, slow, signal)),
    )


def bollinger(
    prices: Any, window: int = 20, n_std: float = 2.0
) -> BollingerBands:
    data = _one_input(prices, "prices")
    window = as_size(window, "window")
    return native_or(
        native_module,
        "rflux_bollinger",
        lambda: BollingerBands(
            *native_module.bollinger(data, window, float(n_std))),
        lambda: BollingerBands(*_multi_output(
            "rflux_bollinger", 3, data.size, get_lib().rflux_bollinger,
            double_ptr(data), data.size, window, float(n_std))),
    )


def atr(
    high: Any, low: Any, close: Any, period: int = 14
) -> np.ndarray:
    high_array = _one_input(high, "high")
    low_array = _one_input(low, "low")
    close_array = _one_input(close, "close")
    if not (high_array.size == low_array.size == close_array.size):
        raise ValueError("high, low and close must have equal lengths")
    period = as_size(period, "period")
    return native_or(
        native_module,
        "rflux_atr",
        lambda: native_module.atr(
            high_array, low_array, close_array, period),
        lambda: call_vector(
            "rflux_atr", high_array.size, get_lib().rflux_atr,
            double_ptr(high_array), double_ptr(low_array),
            double_ptr(close_array), high_array.size, period),
    )


def drawdown(equity: Any) -> np.ndarray:
    return _series_call(
        "rflux_drawdown", equity, input_name="equity")


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
    """Rolling OLS beta of ``asset`` on ``benchmark``; NaN for degenerate windows."""
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

    def fallback() -> Moments:
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

    def native() -> Moments:
        mean, variance, stddev, skewness, kurtosis, valid = (
            native_module.describe(data)
        )
        return Moments(
            float(mean),
            float(variance),
            float(stddev),
            float(skewness),
            float(kurtosis),
            int(valid),
        )

    return native_or(native_module, "rflux_describe", native, fallback)


def quantile(values: Any, q: float) -> float:
    data = _one_input(values)
    return native_or(
        native_module,
        "rflux_quantile",
        lambda: float(native_module.quantile(data, float(q))),
        lambda: call_scalar(
            "rflux_quantile", get_lib().rflux_quantile,
            double_ptr(data), data.size, float(q)),
    )


def pearson(x: Any, y: Any) -> float:
    xa, ya = _pair_inputs(x, y)
    return native_or(
        native_module,
        "rflux_pearson",
        lambda: float(native_module.pearson(xa, ya)),
        lambda: call_scalar(
            "rflux_pearson", get_lib().rflux_pearson,
            double_ptr(xa), double_ptr(ya), xa.size),
    )


def correlation_matrix(panel: Any) -> np.ndarray:
    """Pairwise-complete correlation matrix of a nobs x nassets panel."""
    data = as_f64_2d(panel, "panel")
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("panel must not be empty")
    n_assets = data.shape[1]

    def fallback() -> np.ndarray:
        out = call_vector(
            "rflux_corr_matrix", n_assets * n_assets,
            get_lib().rflux_corr_matrix, double_ptr(data), data.shape[0],
            n_assets)
        return double_matrix(out, n_assets)

    return native_or(
        native_module,
        "rflux_corr_matrix",
        lambda: native_module.correlation_matrix(data),
        fallback,
    )


def _performance_ratio(
    name: str,
    returns_: Any,
    periods_per_year: float,
    risk_free: float,
    kernel: str,
) -> float:
    data = _one_input(returns_, "returns")
    periods_per_year = float(periods_per_year)
    risk_free = float(risk_free)
    return native_or(
        native_module,
        name,
        lambda: float(getattr(native_module, kernel)(
            data, periods_per_year, risk_free)),
        lambda: call_scalar(
            name, getattr(get_lib(), name),
            double_ptr(data), data.size, periods_per_year, risk_free),
    )


def sharpe(
    returns_: Any, periods_per_year: float = 252.0, risk_free: float = 0.0
) -> float:
    return _performance_ratio(
        "rflux_sharpe", returns_, periods_per_year, risk_free, "sharpe")


def sortino(
    returns_: Any, periods_per_year: float = 252.0, risk_free: float = 0.0
) -> float:
    return _performance_ratio(
        "rflux_sortino", returns_, periods_per_year, risk_free, "sortino")


def var_cvar(returns_: Any, confidence: float = 0.95) -> TailRisk:
    """Historical VaR/CVaR; both are positive losses and may be negative."""
    data = _one_input(returns_, "returns")

    def fallback() -> TailRisk:
        var, cvar = _multi_output(
            "rflux_var_cvar", 2, 1, get_lib().rflux_var_cvar,
            double_ptr(data), data.size, float(confidence))
        return TailRisk(float(var[0]), float(cvar[0]))

    return native_or(
        native_module,
        "rflux_var_cvar",
        lambda: TailRisk(*map(float, native_module.var_cvar(
            data, float(confidence)))),
        fallback,
    )


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
    group_count = as_size(group_count, "group_count")
    return native_or(
        native_module,
        "rflux_neutralize",
        lambda: native_module.neutralize(data, group_array, group_count),
        lambda: call_vector(
            "rflux_neutralize", data.size, get_lib().rflux_neutralize,
            double_ptr(data), data.size, int32_ptr(group_array),
            group_count),
    )


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
