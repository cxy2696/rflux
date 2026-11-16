"""High-level, JSON-friendly analysis for a price panel."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

import numpy as np

from . import metrics, percolab, qflux, regimelab
from ._common import asset_names as _asset_names
from ._ffi import abi_version, native_version
from ._json import jsonable
from ._version import __version__


def _as_price_panel(prices: Any) -> np.ndarray:
    array = np.asarray(prices, dtype=np.float64)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError("prices must be a 2D panel with at least 2 rows and 1 asset")
    return np.ascontiguousarray(array)


def _summary(returns: Any, periods_per_year: float, risk_free: float) -> dict[str, Any]:
    values = np.asarray(returns, dtype=np.float64)
    finite = values[np.isfinite(values)]
    result: dict[str, Any] = {
        "n_periods": int(finite.size),
        "elapsed_periods": int(values.size),
        "total_return": None,
        "annualized_return": None,
        "annualized_volatility": None,
        "sharpe": None,
        "sortino": None,
        "max_drawdown": None,
        "calmar": None,
        "var": None,
        "cvar": None,
        "hit_rate": None,
    }
    if finite.size < 2:
        return result
    stats = metrics.performance_summary(
        values,
        periods_per_year=periods_per_year,
        risk_free=risk_free,
    )
    for key, value in asdict(stats).items():
        result[key] = value
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
    panel = _as_price_panel(prices)
    n_obs, n_assets = panel.shape
    asset_names = _asset_names(names, n_assets)
    if not (np.isfinite(periods_per_year) and periods_per_year > 0):
        raise ValueError("periods_per_year must be positive")
    if not np.isfinite(risk_free):
        raise ValueError("risk_free must be finite")
    if include_hmm and hmm_states < 1:
        raise ValueError("hmm_states must be positive")
    if include_network and not 0.0 < network_threshold <= 1.0:
        raise ValueError("network_threshold must be in (0, 1]")

    returns = np.empty_like(panel)
    for column in range(n_assets):
        returns[:, column] = qflux.returns(panel[:, column])

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
            try:
                model = regimelab.fit_hmm(
                    hmm_values.reshape(-1, 1),
                    n_states=hmm_states,
                    max_iter=hmm_max_iter,
                    n_init=hmm_n_init,
                    seed=seed,
                )
                fit = model.fit_info
                filtered = model.filtered(hmm_values.reshape(-1, 1))
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
