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
    as_i32_1d,
    as_size,
    asset_names as _asset_names,
    jsonable,
)
from ._native_backend import native_api as native_module, native_call
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
    """Risk summary and Euler volatility contributions, computed in C++.

    Complete rows are used for the sample covariance.  Weights are exposures
    and may be long-short.
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
    periods_per_year, risk_free, confidence = metrics._validated_options(
        periods_per_year, risk_free, confidence
    )
    assert confidence is not None
    ddof = as_size(ddof, "ddof", minimum=1)
    n_obs = panel.shape[0]
    if int(np.isfinite(panel).all(axis=1).sum()) <= ddof:
        raise ValueError(
            "not enough complete return rows for the requested covariance"
        )
    asset_names = _asset_names(names, n_assets)

    (covariance, asset_volatility, contributions, percentages,
     portfolio_returns, complete_n, annualized_volatility,
     diversification, gross_hhi) = native_call(
        native_module, "rflux_portfolio_risk",
        lambda: native_module.portfolio_risk(
            panel, weight_array, ddof, float(periods_per_year)
        ),
    )
    summary = metrics.performance_summary(
        portfolio_returns,
        periods_per_year=periods_per_year,
        risk_free=risk_free,
        confidence=confidence,
    )
    asset_risk = {
        name: {
            "weight": float(weight),
            "annualized_volatility": float(volatility),
            "component_volatility": (
                None if not np.isfinite(component) else float(component)
            ),
            "percent_contribution": (
                None if not np.isfinite(percent) else float(percent)
            ),
        }
        for name, weight, volatility, component, percent in zip(
            asset_names, weight_array, asset_volatility, contributions,
            percentages
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
            "observations": int(n_obs),
            "complete_observations": int(complete_n),
            "dropped_observations": int(n_obs - complete_n),
            "asset_names": asset_names,
            "weights": [float(value) for value in weight_array],
            "gross_exposure": gross,
            "net_exposure": float(weight_array.sum()),
            "gross_hhi": float(gross_hhi),
        },
        "covariance": covariance.tolist(),
        "asset_risk": asset_risk,
        "portfolio": {
            "annualized_volatility": float(annualized_volatility),
            "diversification_ratio": (
                None if not np.isfinite(diversification)
                else float(diversification)
            ),
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
    contribution, not a compounded contribution.  The statistics kernel runs in
    C++ (``src/regimelab/attribution.cpp``).
    """
    values = as_f64_1d(returns, "returns")
    state_labels = as_i32_1d(labels, "labels")
    if values.size != state_labels.size:
        raise ValueError("returns and labels must have equal lengths")
    if values.size == 0:
        raise ValueError("returns must not be empty")
    if np.any(state_labels < -1):
        raise ValueError("labels must be -1 or non-negative state ids")
    if not np.any(state_labels >= 0):
        raise ValueError("labels must contain at least one valid state")
    periods_per_year, risk_free, confidence = metrics._validated_options(
        periods_per_year, risk_free, confidence
    )
    assert confidence is not None

    payload = native_call(
        native_module, "regime_attribution",
        lambda: native_module.regime_attribution(
            values, state_labels, float(periods_per_year), float(risk_free),
            float(confidence),
        ),
    )
    state_count = int(payload["state_count"])

    def optional(key: str, index: int) -> float | None:
        value = float(payload[key][index])
        return value if math.isfinite(value) else None

    states: dict[str, Any] = {}
    for state in range(state_count):
        states[str(state)] = {
            "periods": int(payload["periods"][state]),
            "period_share": float(payload["period_share"][state]),
            "n_returns": int(payload["n_returns"][state]),
            "mean_return": optional("mean_return", state),
            "annualized_mean_return": optional(
                "annualized_mean_return", state
            ),
            "annualized_volatility": optional(
                "annualized_volatility", state
            ),
            "sharpe": optional("sharpe", state),
            "var": optional("var", state),
            "cvar": optional("cvar", state),
            "hit_rate": optional("hit_rate", state),
            "sum_return": optional("sum_return", state),
            "share_of_total_sum": optional("share_of_total_sum", state),
        }

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
            "labelled_observations": int(payload["labelled_observations"]),
            "states": state_count,
        },
        "states": states,
        "transitions": {
            "counts": payload["transition_counts"].tolist(),
            "probabilities": [
                [
                    None if not math.isfinite(float(value)) else float(value)
                    for value in row
                ]
                for row in payload["transition_probabilities"]
            ],
        },
    }
    return jsonable(report)


__all__ = ["portfolio_risk", "regime_attribution"]
