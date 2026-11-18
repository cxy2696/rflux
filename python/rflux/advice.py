"""Risk-aware research rebalance proposals.

The scoring, allocation, regime and position-sizing kernels are implemented in
C++ (``src/portfoliolab/advice.cpp``); this module validates inputs, drives the
optional HMM step and assembles the JSON-friendly report.
"""

from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Sequence

import numpy as np

from . import metrics, qflux, regimelab
from ._common import (
    as_f64_1d,
    as_f64_panel,
    as_finite_float,
    as_positive_float,
    as_size,
    asset_names as _asset_names,
    jsonable,
    package_metadata,
)
from ._native_backend import native_api as native_module, native_call

__all__ = ["trade_advice"]

_SIGNAL_FIELDS = (
    "last_price",
    "momentum_20",
    "momentum_60",
    "trend",
    "rsi_14",
    "daily_sigma",
    "reference_stop",
)
_ACTIONS = {1: "increase", 0: "hold", -1: "decrease"}


def _optional(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _current_weights(value: Any, n_assets: int) -> np.ndarray | None:
    if value is None:
        return None
    weights = as_f64_1d(value, "current_weights")
    if weights.size != n_assets:
        raise ValueError("current_weights must have one entry per asset")
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("current_weights must be finite and non-negative")
    if float(weights.sum()) > 1.0 + 1e-8:
        raise ValueError("current_weights must sum to at most one")
    return weights.astype(np.float64, copy=True)


def trade_advice(
    prices: Any,
    *,
    names: Sequence[str] | str | None = None,
    current_weights: Any = None,
    periods_per_year: float = 252.0,
    risk_free: float = 0.0,
    max_weight: float = 0.35,
    target_volatility: float = 0.15,
    confidence: float = 0.95,
    portfolio_value: float | None = None,
    risk_per_trade: float = 0.005,
    include_hmm: bool = True,
    hmm_states: int = 3,
    hmm_max_iter: int = 200,
    hmm_n_init: int = 3,
    seed: int = 42,
) -> dict[str, Any]:
    """Build a transparent, long-only rebalance proposal.

    Steps:
      1. HRP gives a covariance-based base allocation.
      2. Cross-sectional momentum/trend and RSI adjust the base weights.
      3. The proposed portfolio's ex-ante volatility and an optional HMM
         stress state scale exposure.
      4. The result is capped, converted to actions and reported with risk.

    The function never shorts, never uses leverage and never claims a
    forecasted return.  Turnover, costs, taxes and execution constraints are
    reported or left to the caller.
    """
    panel = as_f64_panel(prices, "prices", min_rows=3)
    n_obs, n_assets = panel.shape
    asset_names = _asset_names(names, n_assets)
    periods_per_year, risk_free, confidence = metrics._validated_options(
        periods_per_year, risk_free, confidence
    )
    assert confidence is not None
    max_weight = as_finite_float(max_weight, "max_weight")
    if not 0.0 < max_weight <= 1.0:
        raise ValueError("max_weight must be in (0, 1]")
    if max_weight * n_assets < 1.0 - 1e-8:
        raise ValueError("max_weight is too small for the number of assets")
    target_volatility = as_positive_float(target_volatility, "target_volatility")
    if portfolio_value is not None:
        portfolio_value = as_positive_float(portfolio_value, "portfolio_value")
    risk_per_trade = as_finite_float(risk_per_trade, "risk_per_trade")
    if not 0.0 < risk_per_trade < 1.0:
        raise ValueError("risk_per_trade must be in (0, 1)")
    current = _current_weights(current_weights, n_assets)

    returns = qflux.returns_panel(panel)
    market_returns = native_call(
        native_module, "equal_weight_market_returns",
        lambda: native_module.equal_weight_market_returns(returns),
    )
    signal_matrix, scores = native_call(
        native_module, "advice_signals",
        lambda: native_module.advice_signals(panel, returns),
    )
    signals = {
        name: {
            field: _optional(signal_matrix[index, column])
            for column, field in enumerate(_SIGNAL_FIELDS)
        }
        for index, name in enumerate(asset_names)
    }

    covariance, _ = native_call(
        native_module, "complete_case_covariance",
        lambda: native_module.complete_case_covariance(returns, 1),
    )

    regime: dict[str, Any] | None = None
    regime_scale = 1.0
    errors: list[str] = []
    if include_hmm:
        values = native_call(
            native_module, "finite_values",
            lambda: native_module.finite_values(market_returns),
        )
        if values.size >= max(2, hmm_states):
            try:
                snapshot = regimelab.fit_hmm_snapshot(
                    values,
                    n_states=hmm_states,
                    max_iter=hmm_max_iter,
                    n_init=hmm_n_init,
                    seed=seed,
                    horizon=1,
                )
                latest_state, expected_next, stress_probability, scale = (
                    native_call(
                        native_module, "advice_regime",
                        lambda: native_module.advice_regime(
                            snapshot.latest_filtered,
                            snapshot.forecast,
                            snapshot.means[:, 0],
                        ),
                    )
                )
                regime_scale = float(scale)
                regime = {
                    "states": snapshot.n_states,
                    "latest_state": int(latest_state),
                    "latest_filtered": snapshot.latest_filtered.tolist(),
                    "state_means": snapshot.means[:, 0].tolist(),
                    "expected_next_return": float(expected_next),
                    "stress_probability": float(stress_probability),
                    "risk_off": regime_scale < 1.0,
                    "exposure_scale": regime_scale,
                }
            except Exception as exc:
                errors.append(f"HMM skipped: {exc}")
        else:
            errors.append("HMM skipped: not enough finite market returns")

    (
        base, current_resolved, target, deltas, actions,
        market_vol, portfolio_vol, vol_scale, exposure, cash,
        current_cash, turnover,
    ) = native_call(
        native_module, "advice_allocation",
        lambda: native_module.advice_allocation(
            covariance, scores, current, market_returns,
            float(periods_per_year), float(max_weight),
            float(target_volatility), float(regime_scale),
        ),
    )
    current_source = "equal_weight" if current is None else "provided"

    sizing = native_call(
        native_module, "advice_sizing",
        lambda: native_module.advice_sizing(
            target,
            [signals[name]["last_price"] if signals[name]["last_price"] is not None else math.nan
             for name in asset_names],
            [signals[name]["reference_stop"] if signals[name]["reference_stop"] is not None else math.nan
             for name in asset_names],
            portfolio_value, float(risk_per_trade),
        ),
    )
    notional, stop_distance, target_shares, risk_shares, recommended = sizing

    report_actions = []
    for index, name in enumerate(asset_names):
        signal = signals[name]
        reasons = []
        if signal["momentum_20"] is not None:
            reasons.append(f"20d momentum {signal['momentum_20'] * 100:+.1f}%")
        if signal["momentum_60"] is not None:
            reasons.append(f"60d momentum {signal['momentum_60'] * 100:+.1f}%")
        reasons.append(f"trend {signal['trend']:+.1f}")
        if signal["rsi_14"] is not None:
            reasons.append(f"RSI14 {signal['rsi_14']:.1f}")
        report_actions.append({
            "asset": name,
            "action": _ACTIONS[int(actions[index])],
            "current_weight": float(current_resolved[index]),
            "target_weight": float(target[index]),
            "delta": float(deltas[index]),
            "signal_score": float(scores[index]),
            "last_price": signal["last_price"],
            "reference_stop": signal["reference_stop"],
            "stop_distance_pct": _optional(stop_distance[index]),
            "target_notional": _optional(notional[index]),
            "target_shares": (
                int(target_shares[index]) if target_shares[index] >= 0 else None
            ),
            "risk_limited_shares": (
                int(risk_shares[index]) if risk_shares[index] >= 0 else None
            ),
            "recommended_shares": (
                int(recommended[index]) if recommended[index] >= 0 else None
            ),
            "reason": "; ".join(reasons),
        })

    target_series = native_call(
        native_module, "finite_values",
        lambda: native_module.finite_values(
            native_module.portfolio_series(returns, target)
        ),
    )
    current_series = native_call(
        native_module, "finite_values",
        lambda: native_module.finite_values(
            native_module.portfolio_series(returns, current_resolved)
        ),
    )
    target_metrics = (
        metrics.performance_summary(
            target_series,
            periods_per_year=periods_per_year,
            risk_free=risk_free,
        )
        if target_series.size >= 2
        else None
    )
    current_metrics = (
        metrics.performance_summary(
            current_series,
            periods_per_year=periods_per_year,
            risk_free=risk_free,
        )
        if current_series.size >= 2
        else None
    )
    var_cvar = (
        qflux.var_cvar(target_series, confidence)
        if target_series.size >= 2
        else None
    )
    report = {
        "package": package_metadata(),
        "input": {
            "observations": int(n_obs),
            "assets": int(n_assets),
            "asset_names": asset_names,
            "current_weights_source": current_source,
        },
        "signals": signals,
        "allocation": {
            "base_weights": [float(value) for value in base],
            "target_weights": [float(value) for value in target],
            "cash_weight": float(cash),
            "exposure": float(exposure),
            "volatility_scale": float(vol_scale),
            "regime_scale": float(regime_scale),
            "max_weight": max_weight,
            "expected_turnover": float(turnover),
            "actions": report_actions,
        },
        "regime": regime,
        "risk": {
            "market_annualized_volatility": _optional(market_vol),
            "portfolio_annualized_volatility": _optional(portfolio_vol),
            "target_annualized_volatility": (
                target_metrics.annualized_volatility
                if target_metrics is not None else None
            ),
            "target_max_drawdown": (
                target_metrics.max_drawdown if target_metrics is not None else None
            ),
            "target_var": float(var_cvar.var) if var_cvar is not None else None,
            "target_cvar": float(var_cvar.cvar) if var_cvar is not None else None,
            "portfolio_value": portfolio_value,
            "risk_per_trade": risk_per_trade,
            "current": asdict(current_metrics) if current_metrics is not None else None,
            "target": asdict(target_metrics) if target_metrics is not None else None,
        },
        "errors": errors,
        "notes": [
            "Long-only, no leverage; cash is the unallocated remainder.",
            "Weights are research proposals from historical data, not a forecast.",
            "Reference stops are 2 daily-sigma below the last close, not live orders.",
            "Transaction costs, taxes, liquidity and lot sizes are not modelled.",
        ],
        "disclaimer": (
            "Not personalised investment advice. Validate with your own risk "
            "limits, data and execution constraints before trading."
        ),
    }
    return jsonable(report)
