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
from ._native_backend import native_module, native_or

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
    return native_or(
        native_module,
        "rflux_factor_momentum",
        lambda: native_module.factor_momentum(prices, window),
        lambda: _c_factor("rflux_factor_momentum", prices, None, window),
    )



def volume_reversal(close: Any, volume: Any, window: int = 20) -> np.ndarray:
    """Negative trailing return amplified by the current volume ratio."""
    prices = _panel(close, "close")
    volumes = _volume_panel(prices, volume)
    window = as_size(window, "window")
    return native_or(
        native_module,
        "rflux_factor_volume_reversal",
        lambda: native_module.factor_volume_reversal(prices, volumes, window),
        lambda: _c_factor(
            "rflux_factor_volume_reversal", prices, volumes, window),
    )



def vwap_deviation(close: Any, volume: Any, window: int = 20) -> np.ndarray:
    """Rolling VWAP deviation using close as the traded price."""
    prices = _panel(close, "close")
    volumes = _volume_panel(prices, volume)
    window = as_size(window, "window")
    return native_or(
        native_module,
        "rflux_factor_vwap_deviation",
        lambda: native_module.factor_vwap_deviation(prices, volumes, window),
        lambda: _c_factor(
            "rflux_factor_vwap_deviation", prices, volumes, window),
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
    return native_or(
        native_module,
        "rflux_factor_amihud",
        lambda: native_module.factor_amihud(
            prices, volumes, window, float(scale)),
        lambda: _c_factor(
            "rflux_factor_amihud", prices, volumes, window, float(scale)),
    )



def information_coefficient(
    factor: Any,
    forward_returns: Any,
    *,
    method: Literal["pearson", "spearman"] = "pearson",
) -> np.ndarray:
    """Cross-sectional IC for each observation row, implemented in C++.

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
    n_obs = values.shape[0]
    n_assets = values.shape[1]
    def fallback() -> np.ndarray:
        out = empty_f64(n_obs)
        check(
            get_lib().rflux_factor_ic(
                double_ptr(values),
                double_ptr(returns),
                n_obs,
                n_assets,
                1 if normalized == "spearman" else 0,
                double_ptr(out),
            ),
            "rflux_factor_ic",
        )
        return out

    return native_or(
        native_module,
        "rflux_factor_ic",
        lambda: native_module.factor_ic(
            values, returns, normalized == "spearman"
        ),
        fallback,
    )


def factor_turnover(weights: Any) -> np.ndarray:
    """One-way turnover ``0.5 * sum(|w_t - w_{t-1}|)``; first row is zero."""
    panel = _panel(weights, "weights")
    if np.any(~np.isfinite(panel)):
        raise ValueError("weights must be finite")
    n_obs, n_assets = panel.shape
    def fallback() -> np.ndarray:
        out = empty_f64(n_obs)
        check(
            get_lib().rflux_factor_turnover(
                double_ptr(panel), n_obs, n_assets, double_ptr(out)
            ),
            "rflux_factor_turnover",
        )
        return out

    return native_or(
        native_module,
        "rflux_factor_turnover",
        lambda: native_module.factor_turnover(panel),
        fallback,
    )
