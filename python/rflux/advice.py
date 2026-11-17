"""Risk-aware research rebalance proposals.

This module combines trend, momentum, RSI, HRP risk budgets and an optional
HMM market regime to produce a transparent target portfolio.  It is a
research tool, not a personalised investment recommendation: the output lists
the inputs, the signal contribution and the risk estimate behind every weight.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Sequence

import numpy as np

from . import metrics, portfoliolab, qflux, regimelab
from ._common import (
    as_f64_1d,
    as_f64_panel,
    as_finite_float,
    as_positive_float,
    as_size,
    asset_names as _asset_names,
)
from ._json import jsonable
from ._ffi import abi_version, native_version
from ._version import __version__

__all__ = ["trade_advice"]


def _zscore(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    result = np.zeros_like(values, dtype=np.float64)
    if finite.sum() < 2:
        return result
    sample = values[finite]
    scale = float(sample.std(ddof=0))
    if scale <= 0.0 or not np.isfinite(scale):
        return result
    result[finite] = (values[finite] - float(sample.mean())) / scale
    return np.clip(result, -3.0, 3.0)


def _cap_weights(weights: np.ndarray, max_weight: float) -> np.ndarray:
    result = np.clip(weights, 0.0, None)
    total = float(result.sum())
    if total <= 0.0 or not np.isfinite(total):
        result = np.full(weights.size, 1.0 / weights.size)
    else:
        result = result / total
    for _ in range(weights.size + 1):
        over = result > max_weight + 1e-15
        if not over.any():
            break
        excess = float((result[over] - max_weight).sum())
        result[over] = max_weight
        free = ~over
        if not free.any():
            break
        free_total = float(result[free].sum())
        if free_total <= 0.0:
            result[free] = excess / float(free.sum())
        else:
            result[free] += excess * result[free] / free_total
    return result



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


def _signal_row(
    prices: np.ndarray,
    returns: np.ndarray,
    column: int,
) -> dict[str, Any]:
    close = prices[:, column]
    finite = close[np.isfinite(close)]
    if finite.size < 2:
        return {
            "last_price": float(finite[-1]) if finite.size else None,
            "momentum_20": None,
            "momentum_60": None,
            "trend": 0.0,
            "rsi_14": None,
            "daily_sigma": None,
            "reference_stop": None,
        }
    last = float(finite[-1])
    def trailing(window: int) -> float:
        if finite.size <= window:
            return float("nan")
        base = float(finite[-window - 1])
        if base == 0.0:
            return float("nan")
        return last / base - 1.0
    momentum_20 = trailing(20)
    momentum_60 = trailing(60)
    sma20 = qflux.sma(close, min(20, close.size))[-1]
    sma60 = qflux.sma(close, min(60, close.size))[-1]
    trend = 0.0
    if np.isfinite(sma20) and sma20 > 0.0:
        trend += 1.0 if last > float(sma20) else -1.0
    if np.isfinite(sma60) and sma60 > 0.0:
        trend += 1.0 if last > float(sma60) else -1.0
    trend /= 2.0
    try:
        rsi_value = float(qflux.rsi(close, 14)[-1])
    except Exception:
        rsi_value = float("nan")
    column_returns = returns[:, column]
    usable = column_returns[np.isfinite(column_returns)]
    daily_sigma = (
        float(np.std(usable[-20:], ddof=1))
        if usable.size >= 2 else float("nan")
    )
    return {
        "last_price": last,
        "momentum_20": float(momentum_20) if np.isfinite(momentum_20) else None,
        "momentum_60": float(momentum_60) if np.isfinite(momentum_60) else None,
        "trend": float(trend),
        "rsi_14": rsi_value if np.isfinite(rsi_value) else None,
        "daily_sigma": daily_sigma if np.isfinite(daily_sigma) else None,
        "reference_stop": (
            last * (1.0 - 2.0 * daily_sigma)
            if np.isfinite(daily_sigma) and daily_sigma > 0.0 else None
        ),
    }


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
      3. Realised volatility and an optional HMM stress state scale exposure.
      4. The result is capped, converted to actions and reported with risk.

    The function never shorts, never uses leverage and never claims a
    forecasted return.  Turnover, costs, taxes and execution constraints are
    reported or left to the caller.
    """
    panel = as_f64_panel(prices, "prices", min_rows=3)
    n_obs, n_assets = panel.shape
    asset_names = _asset_names(names, n_assets)
    periods_per_year = as_positive_float(periods_per_year, "periods_per_year")
    risk_free = as_finite_float(risk_free, "risk_free")
    max_weight = as_finite_float(max_weight, "max_weight")
    if not 0.0 < max_weight <= 1.0:
        raise ValueError("max_weight must be in (0, 1]")
    if max_weight * n_assets < 1.0 - 1e-8:
        raise ValueError("max_weight is too small for the number of assets")
    target_volatility = as_positive_float(target_volatility, "target_volatility")
    confidence = as_finite_float(confidence, "confidence")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if portfolio_value is not None:
        portfolio_value = as_positive_float(portfolio_value, "portfolio_value")
    risk_per_trade = as_finite_float(risk_per_trade, "risk_per_trade")
    if not 0.0 < risk_per_trade < 1.0:
        raise ValueError("risk_per_trade must be in (0, 1)")
    current = _current_weights(current_weights, n_assets)

    returns = qflux.returns_panel(panel)
    finite_counts = np.sum(np.isfinite(returns), axis=1)
    market_returns = np.full(n_obs, np.nan, dtype=np.float64)
    valid = finite_counts > 0
    market_returns[valid] = (
        np.nansum(returns[valid], axis=1) / finite_counts[valid]
    )

    signals = {
        name: _signal_row(panel, returns, column)
        for column, name in enumerate(asset_names)
    }
    momentum_20 = np.array([
        signals[name]["momentum_20"] if signals[name]["momentum_20"] is not None
        else np.nan
        for name in asset_names
    ])
    momentum_60 = np.array([
        signals[name]["momentum_60"] if signals[name]["momentum_60"] is not None
        else np.nan
        for name in asset_names
    ])
    trend = np.array([signals[name]["trend"] for name in asset_names])
    rsi = np.array([
        signals[name]["rsi_14"] if signals[name]["rsi_14"] is not None
        else np.nan
        for name in asset_names
    ])
    score = (
        0.50 * _zscore(momentum_20)
        + 0.30 * _zscore(momentum_60)
        + 0.20 * trend
    )
    # Penalise crowded overbought names and give a small cushion to oversold
    # names.  The size is deliberately modest relative to the trend signal.
    rsi_adjust = np.zeros(n_assets, dtype=np.float64)
    rsi_adjust[np.isfinite(rsi) & (rsi >= 75.0)] = -0.20
    rsi_adjust[np.isfinite(rsi) & (rsi <= 25.0)] = 0.10
    score = np.clip(score + rsi_adjust, -1.0, 1.0)

    covariance = np.cov(returns, rowvar=False)
    covariance = np.atleast_2d(covariance)
    base = None
    if n_assets == 1:
        base = np.array([1.0], dtype=np.float64)
    else:
        try:
            base = portfoliolab.hrp_weights(covariance)
        except Exception:
            base = None
    if base is None:
        diagonal = np.diag(covariance).astype(np.float64, copy=True)
        diagonal[~np.isfinite(diagonal) | (diagonal <= 0.0)] = np.nan
        if np.all(np.isnan(diagonal)):
            base = np.full(n_assets, 1.0 / n_assets)
        else:
            inverse = 1.0 / diagonal
            inverse[~np.isfinite(inverse)] = np.nan
            if not np.any(np.isfinite(inverse)):
                base = np.full(n_assets, 1.0 / n_assets)
            else:
                inverse = np.where(np.isfinite(inverse), inverse, 0.0)
                base = inverse / inverse.sum()
    base = _cap_weights(base, max_weight)

    tilted = base * (1.0 + 0.5 * score)
    if not np.any(tilted > 0.0):
        tilted = base.copy()
    tilted = _cap_weights(tilted, max_weight)

    market_vol = (
        float(np.nanstd(market_returns, ddof=1) * np.sqrt(periods_per_year))
        if np.isfinite(market_returns).sum() > 1
        else float("nan")
    )
    vol_scale = 1.0
    if np.isfinite(market_vol) and market_vol > target_volatility:
        vol_scale = target_volatility / market_vol

    regime: dict[str, Any] | None = None
    regime_scale = 1.0
    errors: list[str] = []
    if include_hmm:
        values = market_returns[np.isfinite(market_returns)]
        if values.size >= max(2, hmm_states):
            model = None
            try:
                model = regimelab.fit_hmm(
                    values.reshape(-1, 1),
                    n_states=hmm_states,
                    max_iter=hmm_max_iter,
                    n_init=hmm_n_init,
                    seed=seed,
                )
                filtered = model.filtered(values.reshape(-1, 1))[-1]
                forecast = model.forecast(filtered, horizon=1)
                means = model.parameters.means[:, 0]
                expected_next = float(np.dot(forecast, means))
                stress_probability = float(
                    filtered[means < 0.0].sum()
                ) if np.any(means < 0.0) else 0.0
                if expected_next < 0.0 and float(filtered.max()) >= 0.5:
                    regime_scale = 0.5
                elif stress_probability >= 0.65:
                    regime_scale = 0.7
                regime = {
                    "states": int(model.n_states),
                    "latest_state": int(np.argmax(filtered)),
                    "latest_filtered": filtered.tolist(),
                    "state_means": means.tolist(),
                    "expected_next_return": expected_next,
                    "stress_probability": stress_probability,
                    "risk_off": regime_scale < 1.0,
                    "exposure_scale": regime_scale,
                }
            except Exception as exc:
                errors.append(f"HMM skipped: {exc}")
            finally:
                if model is not None:
                    model.close()
        else:
            errors.append("HMM skipped: not enough finite market returns")

    exposure = float(np.clip(vol_scale * regime_scale, 0.0, 1.0))
    target = tilted * exposure
    cash = float(max(0.0, 1.0 - target.sum()))
    if current is None:
        current = np.full(n_assets, 1.0 / n_assets, dtype=np.float64)
        current_source = "equal_weight"
    else:
        current_source = "provided"
    delta = target - current
    actions = []
    for index, name in enumerate(asset_names):
        change = float(delta[index])
        action = "hold"
        if change > 0.02:
            action = "increase"
        elif change < -0.02:
            action = "decrease"
        reasons = []
        if signals[name]["momentum_20"] is not None:
            reasons.append(
                f"20d momentum {signals[name]['momentum_20'] * 100:+.1f}%"
            )
        if signals[name]["momentum_60"] is not None:
            reasons.append(
                f"60d momentum {signals[name]['momentum_60'] * 100:+.1f}%"
            )
        reasons.append(f"trend {signals[name]['trend']:+.1f}")
        if signals[name]["rsi_14"] is not None:
            reasons.append(f"RSI14 {signals[name]['rsi_14']:.1f}")
        signal = signals[name]
        stop = signal["reference_stop"]
        last_price = signal["last_price"]
        notional = (
            float(target[index]) * portfolio_value
            if portfolio_value is not None else None
        )
        target_shares = (
            int(np.floor(notional / last_price))
            if notional is not None and last_price and last_price > 0.0
            else None
        )
        risk_shares = None
        if (
            portfolio_value is not None and stop is not None and last_price
            and last_price > stop > 0.0
        ):
            risk_shares = int(np.floor(
                portfolio_value * risk_per_trade / (last_price - stop)
            ))
        recommended_shares = None
        if target_shares is not None:
            recommended_shares = target_shares
            if risk_shares is not None:
                recommended_shares = min(target_shares, risk_shares)
        actions.append({
            "asset": name,
            "action": action,
            "current_weight": float(current[index]),
            "target_weight": float(target[index]),
            "delta": change,
            "signal_score": float(score[index]),
            "last_price": last_price,
            "reference_stop": stop,
            "stop_distance_pct": (
                float((last_price - stop) / last_price)
                if last_price and stop is not None and last_price > 0.0
                else None
            ),
            "target_notional": notional,
            "target_shares": target_shares,
            "risk_limited_shares": risk_shares,
            "recommended_shares": recommended_shares,
            "reason": "; ".join(reasons),
        })

    target_series = np.nansum(returns * target, axis=1)
    current_series = np.nansum(returns * current, axis=1)
    target_series = target_series[np.isfinite(target_series)]
    current_series = current_series[np.isfinite(current_series)]
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
    turnover = float(0.5 * np.abs(delta).sum())
    report = {
        "package": {
            "version": __version__,
            "native_version": native_version(),
            "abi_version": abi_version(),
        },
        "input": {
            "observations": int(n_obs),
            "assets": int(n_assets),
            "asset_names": asset_names,
            "current_weights_source": current_source,
        },
        "signals": signals,
        "allocation": {
            "base_weights": base.tolist(),
            "target_weights": target.tolist(),
            "cash_weight": cash,
            "exposure": exposure,
            "volatility_scale": vol_scale,
            "regime_scale": regime_scale,
            "max_weight": max_weight,
            "expected_turnover": turnover,
            "actions": actions,
        },
        "regime": regime,
        "risk": {
            "market_annualized_volatility": market_vol,
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
