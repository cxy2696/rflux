"""High-level, JSON-friendly analysis for a price panel."""

from __future__ import annotations

from dataclasses import asdict, fields
from typing import Any, Sequence

import numpy as np

from . import metrics, percolab, qflux, regimelab
from ._common import (
    as_f64_panel,
    as_finite_float,
    as_positive_float,
    as_size,
    asset_names as _asset_names,
)
from ._ffi import abi_version, native_version
from ._json import jsonable
from ._version import __version__


_SUMMARY_FIELDS = tuple(field.name for field in fields(metrics.PerformanceMetrics))


def _summary(returns: Any, periods_per_year: float, risk_free: float) -> dict[str, Any]:
    values = np.asarray(returns, dtype=np.float64)
    finite = values[np.isfinite(values)]
    result: dict[str, Any] = {field: None for field in _SUMMARY_FIELDS}
    result["n_periods"] = int(finite.size)
    result["elapsed_periods"] = int(values.size)
    if finite.size < 2:
        return result
    stats = metrics.performance_summary(
        values,
        periods_per_year=periods_per_year,
        risk_free=risk_free,
    )
    result.update(asdict(stats))
    return result


def analyze_prices(
    prices: Any,
    *,
    names: Sequence[str] | str | None = None,
    periods_per_year: float = 252.0,
    risk_free: float = 0.0,
    include_hmm: bool = True,
    hmm_states: int = 3,
    hmm_max_iter: int = 200,
    hmm_n_init: int = 3,
    include_network: bool = True,
    network_threshold: float = 0.5,
    include_correlation_matrix: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Summarize a price panel without pandas.

    Parameters are intentionally explicit. Prices are level series with shape
    ``(observations, assets)``; a 1D array is treated as one asset. The return
    value contains plain Python values and can be passed directly to
    ``json.dumps`` after replacing NaN with None, which this function already
    does.

    Optional HMM and network steps are reported under their own keys. Failures
    in those sections are listed under ``errors`` and do not hide the basic
    return statistics.
    """
    panel = as_f64_panel(prices, "prices", min_rows=2)
    n_obs, n_assets = panel.shape
    asset_names = _asset_names(names, n_assets)
    periods_per_year = as_positive_float(periods_per_year, "periods_per_year")
    risk_free = as_finite_float(risk_free, "risk_free")
    if include_hmm:
        hmm_states = as_size(hmm_states, "hmm_states")
        hmm_max_iter = as_size(hmm_max_iter, "hmm_max_iter")
        hmm_n_init = as_size(hmm_n_init, "hmm_n_init")
    if include_network and not 0.0 < network_threshold <= 1.0:
        raise ValueError("network_threshold must be in (0, 1]")

    returns = qflux.returns_panel(panel)

    counts = np.sum(np.isfinite(returns), axis=1)
    sums = np.nansum(returns, axis=1)
    market_returns = np.full(n_obs, np.nan)
    valid_rows = counts > 0
    market_returns[valid_rows] = sums[valid_rows] / counts[valid_rows]

    asset_stats: dict[str, Any] = {}
    for column, name in enumerate(asset_names):
        finite_prices = panel[np.isfinite(panel[:, column]), column]
        item = _summary(returns[:, column], periods_per_year, risk_free)
        item["last_price"] = (
            float(finite_prices[-1]) if finite_prices.size else None
        )
        try:
            rsi_value = float(qflux.rsi(panel[:, column], 14)[-1])
        except Exception:
            rsi_value = float("nan")
        item["rsi_14"] = rsi_value
        asset_stats[name] = item

    errors: list[str] = []
    market = _summary(market_returns, periods_per_year, risk_free)
    report: dict[str, Any] = {
        "package": {
            "version": __version__,
            "native_version": native_version(),
            "abi_version": abi_version(),
        },
        "input": {
            "observations": int(n_obs),
            "assets": int(n_assets),
            "asset_names": asset_names,
        },
        "asset_stats": asset_stats,
        "market": market,
        "correlation": {},
        "network": None,
        "hmm": None,
        "errors": errors,
    }

    correlation = qflux.correlation_matrix(returns)
    finite_corr = correlation[np.isfinite(correlation)]
    report["correlation"] = {
        "minimum": float(finite_corr.min()) if finite_corr.size else None,
        "maximum": float(finite_corr.max()) if finite_corr.size else None,
        "mean": float(finite_corr.mean()) if finite_corr.size else None,
    }
    if include_correlation_matrix:
        report["correlation"]["matrix"] = correlation.tolist()

    if include_network and n_assets >= 2:
        if finite_corr.size != n_assets * n_assets:
            errors.append(
                "network skipped: correlation matrix contains undefined entries"
            )
        else:
            try:
                network = percolab.network_from_correlation(
                    correlation, network_threshold
                )
                order = np.argsort(network.eigenvector)[::-1]
                report["network"] = {
                    "threshold": float(network_threshold),
                    "edges": int(network.n_edges),
                    "density": float(network.density),
                    "components": int(network.components.n_components),
                    "largest_component": int(network.components.largest_size),
                    "average_clustering": float(network.average_clustering),
                    "transitivity": float(network.transitivity),
                    "assortativity": float(network.assortativity),
                    "efficiency": float(network.path_stats.efficiency),
                    "average_path_length": float(
                        network.path_stats.average_path_length
                    ),
                    "diameter": int(network.path_stats.diameter),
                    "connected": bool(network.path_stats.connected),
                    "max_core": int(network.max_core),
                    "mst_total_weight": float(network.mst.weight.sum()),
                    "top_eigenvector": [
                        {
                            "asset": asset_names[int(index)],
                            "value": float(network.eigenvector[int(index)]),
                        }
                        for index in order[: min(5, n_assets)]
                    ],
                }
            except Exception as exc:
                errors.append(f"network skipped: {exc}")

    if include_hmm:
        hmm_values = market_returns[np.isfinite(market_returns)]
        if hmm_values.size < max(2, hmm_states):
            errors.append("HMM skipped: not enough finite market returns")
        else:
            model = None
            observations = hmm_values.reshape(-1, 1)
            try:
                model = regimelab.fit_hmm(
                    observations,
                    n_states=hmm_states,
                    max_iter=hmm_max_iter,
                    n_init=hmm_n_init,
                    seed=seed,
                )
                fit = model.fit_info
                filtered = model.filtered(observations)
                latest = filtered[-1]
                forecast = model.forecast(latest, horizon=5)
                parameters = model.parameters
                report["hmm"] = {
                    "states": int(model.n_states),
                    "iterations": int(fit.iterations),
                    "converged": bool(fit.converged),
                    "log_likelihood": float(fit.log_likelihood),
                    "aic": float(fit.aic),
                    "bic": float(fit.bic),
                    "latest_state": int(np.argmax(latest)),
                    "latest_filtered": latest.tolist(),
                    "forecast_5": forecast.tolist(),
                    "transition": parameters.transition.tolist(),
                    "state_means": parameters.means[:, 0].tolist(),
                }
            except Exception as exc:
                errors.append(f"HMM skipped: {exc}")
            finally:
                if model is not None:
                    model.close()

    return jsonable(report)


__all__ = ["analyze_prices"]
