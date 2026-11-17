"""Factor engineering and evaluation.

The C++ core supplies the panel kernels (momentum, volume reversal, VWAP
deviation and Amihud illiquidity).  When the optional pybind11 extension is
absent, the same kernels are called through the stable C ABI.  Factor panels
are always ``(n_obs, n_assets)`` with NaN for warm-up or missing input cells.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ._common import as_f64_2d, as_size, double_ptr, empty_f64
from ._ffi import check, get_lib
from ._native_backend import native_module, raise_as_rflux
from . import qflux

__all__ = [
    "amihud",
    "factor_turnover",
    "information_coefficient",
    "momentum",
    "volume_reversal",
    "vwap_deviation",
]


def _panel(values: Any, name: str = "panel") -> np.ndarray:
    array = as_f64_2d(values, name)
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    return array


def _volume_panel(close: np.ndarray, volume: Any) -> np.ndarray:
    values = _panel(volume, "volume")
    if values.shape != close.shape:
        raise ValueError("close and volume panels must have the same shape")
    if np.any(values < 0.0):
        raise ValueError("volume must be non-negative")
    return values



def _c_factor(
    name: str,
    prices: np.ndarray,
    volumes: np.ndarray | None,
    window: int,
    scale: float | None = None,
) -> np.ndarray:
    if window <= 0:
        raise ValueError("window must be positive")
    n_obs, n_assets = prices.shape
    out = empty_f64(n_obs * n_assets)
    function = getattr(get_lib(), name)
    if volumes is None:
        check(
            function(
                double_ptr(prices), n_obs, n_assets, window, double_ptr(out)
            ),
            name,
        )
    elif scale is None:
        check(
            function(
                double_ptr(prices),
                double_ptr(volumes),
                n_obs,
                n_assets,
                window,
                double_ptr(out),
            ),
            name,
        )
    else:
        check(
            function(
                double_ptr(prices),
                double_ptr(volumes),
                n_obs,
                n_assets,
                window,
                float(scale),
                double_ptr(out),
            ),
            name,
        )
    return out.reshape(n_obs, n_assets)


def momentum(close: Any, window: int = 20) -> np.ndarray:
    """Trailing simple return ``close[t] / close[t-window] - 1``."""
    prices = _panel(close, "close")
    window = as_size(window, "window")
    if native_module is not None:
        try:
            return native_module.factor_momentum(prices, window)
        except native_module.NativeError as exc:
            raise_as_rflux("rflux_factor_momentum", exc)
    return _c_factor("rflux_factor_momentum", prices, None, window)



def volume_reversal(close: Any, volume: Any, window: int = 20) -> np.ndarray:
    """Negative trailing return amplified by the current volume ratio."""
    prices = _panel(close, "close")
    volumes = _volume_panel(prices, volume)
    window = as_size(window, "window")
    if native_module is not None:
        try:
            return native_module.factor_volume_reversal(
                prices, volumes, window
            )
        except native_module.NativeError as exc:
            raise_as_rflux("rflux_factor_volume_reversal", exc)
    return _c_factor(
        "rflux_factor_volume_reversal", prices, volumes, window
    )



def vwap_deviation(close: Any, volume: Any, window: int = 20) -> np.ndarray:
    """Rolling VWAP deviation using close as the traded price."""
    prices = _panel(close, "close")
    volumes = _volume_panel(prices, volume)
    window = as_size(window, "window")
    if native_module is not None:
        try:
            return native_module.factor_vwap_deviation(
                prices, volumes, window
            )
        except native_module.NativeError as exc:
            raise_as_rflux("rflux_factor_vwap_deviation", exc)
    return _c_factor(
        "rflux_factor_vwap_deviation", prices, volumes, window
    )



def amihud(
    close: Any, volume: Any, window: int = 20, scale: float = 1e6
) -> np.ndarray:
    """Rolling Amihud illiquidity, scaled by ``scale``."""
    prices = _panel(close, "close")
    volumes = _volume_panel(prices, volume)
    window = as_size(window, "window")
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    if native_module is not None:
        try:
            return native_module.factor_amihud(
                prices, volumes, window, float(scale)
            )
        except native_module.NativeError as exc:
            raise_as_rflux("rflux_factor_amihud", exc)
    return _c_factor(
        "rflux_factor_amihud", prices, volumes, window, float(scale)
    )



def information_coefficient(
    factor: Any,
    forward_returns: Any,
    *,
    method: Literal["pearson", "spearman"] = "pearson",
) -> np.ndarray:
    """Cross-sectional IC for each observation row.

    Rows with fewer than two valid assets or a zero-dispersion side yield NaN.
    ``spearman`` computes Pearson correlation of average ranks.
    """
    values = _panel(factor, "factor")
    returns = _panel(forward_returns, "forward_returns")
    if values.shape != returns.shape:
        raise ValueError("factor and forward_returns must have the same shape")
    normalized = str(method).lower()
    if normalized not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    for row in range(values.shape[0]):
        x = values[row]
        y = returns[row]
        valid = np.isfinite(x) & np.isfinite(y)
        if int(valid.sum()) < 2:
            continue
        a = x[valid]
        b = y[valid]
        if normalized == "spearman":
            a = qflux.rank(a)
            b = qflux.rank(b)
        a64 = np.asarray(a, dtype=np.float64)
        b64 = np.asarray(b, dtype=np.float64)
        if np.all(a64 == a64[0]) or np.all(b64 == b64[0]):
            continue
        result[row] = float(np.corrcoef(a64, b64)[0, 1])
    return result


def factor_turnover(weights: Any) -> np.ndarray:
    """One-way turnover ``0.5 * sum(|w_t - w_{t-1}|)``; first row is zero."""
    panel = _panel(weights, "weights")
    if np.any(~np.isfinite(panel)):
        raise ValueError("weights must be finite")
    result = np.zeros(panel.shape[0], dtype=np.float64)
    for t in range(1, panel.shape[0]):
        result[t] = 0.5 * float(np.abs(panel[t] - panel[t - 1]).sum())
    return result
