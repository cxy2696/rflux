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

if _native is not None:
    GaussianHMM = _native.GaussianHMM
    NativeError = _native.NativeError
    atr = _native.atr
    bollinger = _native.bollinger
    correlation_matrix = _native.correlation_matrix
    describe = _native.describe
    factor_amihud = _native.factor_amihud
    factor_momentum = _native.factor_momentum
    factor_volume_reversal = _native.factor_volume_reversal
    factor_vwap_deviation = _native.factor_vwap_deviation
    hrp_weights = _native.hrp_weights
    huber_regression = _native.huber_regression
    almgren_chriss = _native.almgren_chriss
    macd = _native.macd
    neutralize = _native.neutralize
    pair_series = _native.pair_series
    pearson = _native.pearson
    quantile = _native.quantile
    series = _native.series
    series_into = _native.series_into
    sharpe = _native.sharpe
    sortino = _native.sortino
    var_cvar = _native.var_cvar
    __all__ += [
        "GaussianHMM",
        "NativeError",
        "atr",
        "bollinger",
        "correlation_matrix",
        "almgren_chriss",
        "describe",
        "factor_amihud",
        "factor_momentum",
        "factor_volume_reversal",
        "factor_vwap_deviation",
        "hrp_weights",
        "huber_regression",
        "macd",
        "neutralize",
        "pair_series",
        "pearson",
        "quantile",
        "series",
        "series_into",
        "sharpe",
        "sortino",
        "var_cvar",
    ]
