"""Application-level portfolio and regime diagnostics.

The functions here are thin: they turn a return panel and a
label series into JSON-friendly reports using the same qflux kernels as the
rest of the package.  The formulas and assumptions are documented in
``docs/APPLICATIONS_cn.md`` and ``docs/APPLICATIONS_en.md``.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Sequence

import numpy as np

from . import metrics, qflux
from ._common import (
    as_f64_1d,
    as_f64_panel,
    as_finite_float,
    as_i32_1d,
    as_open_unit_float,
    as_positive_float,
    as_size,
    asset_names as _asset_names,
)
from ._json import jsonable
from ._version import __version__


def portfolio_risk(
    returns: Any,
    weights: Any,
    *,
    names: Sequence[str] | str | None = None,
    periods_per_year: float = 252.0,
    risk_free: float = 0.0,
    confidence: float = 0.95,
    ddof: int = 1,
) -> dict[str, Any]:
    """Risk summary and Euler volatility contributions for a return panel.

    ``returns`` has shape ``(observations, assets)`` and contains periodic
    arithmetic returns.  Rows with any missing return are removed before the
    covariance is estimated, so every asset uses the same sample.  Weights are
    exposures per unit of capital; they may be long-short and need not sum to
    one.  The portfolio return is ``returns_complete @ weights``.

    The covariance uses ``ddof`` (default 1).  The volatility contributions
    satisfy ``sum(component_volatility) == annualized_volatility`` whenever
    portfolio volatility is positive.
    """
    panel = as_f64_panel(returns, "returns", min_rows=2)
    n_assets = panel.shape[1]
    weight_array = as_f64_1d(weights, "weights")
    if weight_array.size != n_assets:
        raise ValueError(f"weights must have length {n_assets}")
    if not np.all(np.isfinite(weight_array)):
        raise ValueError("weights must be finite")
    gross = float(np.sum(np.abs(weight_array)))
    if not (gross > 0.0):
        raise ValueError("weights must have positive gross exposure")
    periods_per_year = as_positive_float(periods_per_year, "periods_per_year")
    risk_free = as_finite_float(risk_free, "risk_free")
    confidence = as_open_unit_float(confidence, "confidence")
    ddof = as_size(ddof, "ddof", minimum=1)

    complete = panel[np.all(np.isfinite(panel), axis=1)]
    complete_n = int(complete.shape[0])
    if complete_n <= ddof:
        raise ValueError(
            "not enough complete return rows for the requested covariance"
        )
    asset_names = _asset_names(names, n_assets)
    centered = complete - complete.mean(axis=0)
    covariance = centered.T @ centered / float(complete_n - ddof)
    portfolio_returns = complete @ weight_array
    portfolio_variance = float(weight_array @ covariance @ weight_array)
    if portfolio_variance < 0.0 and portfolio_variance > -1e-18:
        portfolio_variance = 0.0
    if portfolio_variance < 0.0:
        raise ValueError("portfolio covariance produced a negative variance")
    annualized_volatility = math.sqrt(portfolio_variance * periods_per_year)

    asset_variance = np.diag(covariance)
    asset_volatility = np.sqrt(np.maximum(asset_variance, 0.0) * periods_per_year)
    contributions: list[float | None]
    percentages: list[float | None]
    if annualized_volatility > 0.0:
        marginal = covariance @ weight_array / math.sqrt(portfolio_variance)
        component = weight_array * marginal * math.sqrt(periods_per_year)
        contributions = [float(value) for value in component]
        percentages = [float(value / annualized_volatility) for value in component]
    else:
        contributions = [None] * n_assets
        percentages = [None] * n_assets

    summary = metrics.performance_summary(
        portfolio_returns,
        periods_per_year=periods_per_year,
        risk_free=risk_free,
        confidence=confidence,
    )
    diversification = (
        float(np.abs(weight_array) @ asset_volatility / annualized_volatility)
        if annualized_volatility > 0.0
        else None
    )
    gross_hhi = float(np.sum((weight_array / gross) ** 2))
    asset_risk = {
        name: {
            "weight": float(weight),
            "annualized_volatility": float(volatility),
            "component_volatility": contributions[index],
            "percent_contribution": percentages[index],
        }
        for index, (name, weight, volatility) in enumerate(
            zip(asset_names, weight_array, asset_volatility)
        )
    }

    report = {
        "package": {"version": __version__},
        "method": {
            "estimator": "sample_covariance_complete_cases",
            "ddof": int(ddof),
            "periods_per_year": float(periods_per_year),
            "annualization": "sqrt(periods_per_year)",
            "metrics_ddof": 1,
            "risk_free": float(risk_free),
            "confidence": float(confidence),
            "contribution": "Euler volatility contribution",
        },
        "input": {
            "observations": int(panel.shape[0]),
            "complete_observations": complete_n,
            "dropped_observations": int(panel.shape[0] - complete_n),
            "asset_names": asset_names,
            "weights": [float(value) for value in weight_array],
            "gross_exposure": gross,
            "net_exposure": float(weight_array.sum()),
            "gross_hhi": gross_hhi,
        },
        "covariance": covariance.tolist(),
        "asset_risk": asset_risk,
        "portfolio": {
            "annualized_volatility": annualized_volatility,
            "diversification_ratio": diversification,
            "metrics": asdict(summary),
        },
    }
    return jsonable(report)


def regime_attribution(
    returns: Any,
    labels: Any,
    *,
    periods_per_year: float = 252.0,
    risk_free: float = 0.0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Summarize periodic returns by an existing state-label series.

    Labels must be ``-1`` (unlabelled/warm-up) or non-negative state ids.
    Consecutive valid labels form the transition count matrix.  State
    statistics use only finite returns; ``share_of_total_sum`` is an arithmetic
    contribution, not a compounded contribution.
    """
    values = as_f64_1d(returns, "returns")
    state_labels = as_i32_1d(labels, "labels")
    if values.size != state_labels.size:
        raise ValueError("returns and labels must have equal lengths")
    if values.size == 0:
        raise ValueError("returns must not be empty")
    if np.any(state_labels < -1):
        raise ValueError("labels must be -1 or non-negative state ids")
    if np.any(state_labels >= 0):
        state_count = int(state_labels.max()) + 1
    else:
        raise ValueError("labels must contain at least one valid state")
    periods_per_year = as_positive_float(periods_per_year, "periods_per_year")
    risk_free = as_finite_float(risk_free, "risk_free")
    confidence = as_open_unit_float(confidence, "confidence")

    total_sum = float(np.sum(values[np.isfinite(values)]))
    states: dict[str, Any] = {}
    for state in range(state_count):
        mask = state_labels == state
        selected = values[mask & np.isfinite(values)]
        count = int(selected.size)
        state_periods = int(mask.sum())
        entry: dict[str, Any] = {
            "periods": state_periods,
            "period_share": float(state_periods / values.size),
            "n_returns": count,
            "mean_return": None,
            "annualized_mean_return": None,
            "annualized_volatility": None,
            "sharpe": None,
            "var": None,
            "cvar": None,
            "hit_rate": None,
            "sum_return": None,
            "share_of_total_sum": None,
        }
        if count:
            mean = float(selected.mean())
            entry["mean_return"] = mean
            entry["annualized_mean_return"] = mean * periods_per_year
            entry["hit_rate"] = float(np.mean(selected > 0.0))
            entry["sum_return"] = float(selected.sum())
            if total_sum != 0.0:
                entry["share_of_total_sum"] = float(
                    selected.sum() / total_sum
                )
            tail = qflux.var_cvar(selected, confidence)
            entry["var"] = float(tail.var)
            entry["cvar"] = float(tail.cvar)
            if count >= 2:
                entry["annualized_volatility"] = float(
                    selected.std(ddof=1) * math.sqrt(periods_per_year)
                )
                entry["sharpe"] = float(
                    qflux.sharpe(selected, periods_per_year, risk_free)
                )
        states[str(state)] = entry

    counts = np.zeros((state_count, state_count), dtype=np.int64)
    for first, second in zip(state_labels[:-1], state_labels[1:]):
        if first >= 0 and second >= 0:
            counts[int(first), int(second)] += 1
    row_sums = counts.sum(axis=1)
    probabilities: list[list[float | None]] = []
    for row in range(state_count):
        if row_sums[row] == 0:
            probabilities.append([None] * state_count)
        else:
            probabilities.append(
                [float(value / row_sums[row]) for value in counts[row]]
            )

    report = {
        "package": {"version": __version__},
        "method": {
            "labels": "supplied state ids; -1 ignored",
            "returns": "periodic arithmetic returns",
            "annualization": "mean * periods_per_year; std * sqrt(periods_per_year)",
            "transition": "counts of consecutive valid labels",
            "risk_free": float(risk_free),
            "confidence": float(confidence),
        },
        "input": {
            "observations": int(values.size),
            "labelled_observations": int(np.count_nonzero(state_labels >= 0)),
            "states": state_count,
        },
        "states": states,
        "transitions": {
            "counts": counts.tolist(),
            "probabilities": probabilities,
        },
    }
    return jsonable(report)


__all__ = ["portfolio_risk", "regime_attribution"]
