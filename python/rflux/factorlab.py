"""Factor engineering and evaluation.

The C++ core supplies the panel kernels (momentum, volume reversal, VWAP
deviation and Amihud illiquidity).  Factor panels are always
``(n_obs, n_assets)`` with NaN for warm-up or missing input cells.
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from ._common import as_f64_2d, as_size
from ._native_backend import native_api as native_module

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


def momentum(close: Any, window: int = 20) -> np.ndarray:
    """Trailing simple return ``close[t] / close[t-window] - 1``."""
    return native_module.factor_momentum(
        _panel(close, "close"), as_size(window, "window"))


def volume_reversal(close: Any, volume: Any, window: int = 20) -> np.ndarray:
    """Negative trailing return amplified by the current volume ratio."""
    prices = _panel(close, "close")
    return native_module.factor_volume_reversal(
        prices, _volume_panel(prices, volume), as_size(window, "window"))


def vwap_deviation(close: Any, volume: Any, window: int = 20) -> np.ndarray:
    """Rolling VWAP deviation using close as the traded price."""
    prices = _panel(close, "close")
    return native_module.factor_vwap_deviation(
        prices, _volume_panel(prices, volume), as_size(window, "window"))


def amihud(
    close: Any, volume: Any, window: int = 20, scale: float = 1e6
) -> np.ndarray:
    """Rolling Amihud illiquidity, scaled by ``scale``."""
    prices = _panel(close, "close")
    volumes = _volume_panel(prices, volume)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be finite and positive")
    return native_module.factor_amihud(
        prices, volumes, as_size(window, "window"), float(scale))


def information_coefficient(
    factor: Any,
    forward_returns: Any,
    *,
    method: Literal["pearson", "spearman"] = "pearson",
    min_pairs: int = 2,
) -> np.ndarray:
    """Cross-sectional IC for each observation row, implemented in C++.

    Rows with fewer than ``min_pairs`` valid assets or a zero-dispersion side
    yield NaN.  ``spearman`` computes Pearson correlation of average ranks.
    """
    values = _panel(factor, "factor")
    returns = _panel(forward_returns, "forward_returns")
    if values.shape != returns.shape:
        raise ValueError("factor and forward_returns must have the same shape")
    normalized = str(method).lower()
    if normalized not in {"pearson", "spearman"}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    return native_module.factor_ic(
        values, returns, normalized == "spearman",
        as_size(min_pairs, "min_pairs", minimum=2),
    )


def factor_turnover(weights: Any) -> np.ndarray:
    """One-way turnover ``0.5 * sum(|w_t - w_{t-1}|)``; first row is zero."""
    panel = _panel(weights, "weights")
    if np.any(~np.isfinite(panel)):
        raise ValueError("weights must be finite")
    return native_module.factor_turnover(panel)


__all__ = [
    "amihud",
    "factor_turnover",
    "information_coefficient",
    "momentum",
    "volume_reversal",
    "vwap_deviation",
]
