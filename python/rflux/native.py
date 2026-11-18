"""Optional low-level access to the pybind11 bridge.

Most users should call :mod:`rflux.qflux` and :mod:`rflux.regimelab`; they use
the native bridge automatically.  This module exposes the raw zero-copy entry
points for applications that need to reuse an output buffer or inspect native
availability.  Every attribute is ``None``/absent when the extension was not
built into the installation.
"""

from __future__ import annotations

from ._native_backend import (
    NATIVE_AVAILABLE as available,
    native_info as info,
    native_module as _native,
)

__all__ = ["available", "info"]

_EXPORTS = (
    "GaussianHMM",
    "NativeError",
    "almgren_chriss",
    "atr",
    "bollinger",
    "correlation_matrix",
    "describe",
    "factor_amihud",
    "factor_ic",
    "factor_momentum",
    "factor_volume_reversal",
    "factor_turnover",
    "factor_vwap_deviation",
    "hrp_weights",
    "huber_regression",
    "macd",
    "nco_weights",
    "neutralize",
    "pair_series",
    "pearson",
    "quantile",
    "rebalance_targets",
    "risk_parity_weights",
    "series",
    "series_into",
    "sharpe",
    "sortino",
    "tail_dependence",
    "var_cvar",
)

if _native is not None:
    # Bind the public surface in one place so a new native entry point cannot
    # be exported in one list but forgotten in the other.
    globals().update({name: getattr(_native, name) for name in _EXPORTS})
    __all__ += list(_EXPORTS)
