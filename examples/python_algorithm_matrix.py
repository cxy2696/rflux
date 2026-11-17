"""Run one example from each algorithm-matrix module."""

from __future__ import annotations

import numpy as np

import rflux


def main() -> None:
    rng = np.random.default_rng(7)
    n_obs, n_assets = 320, 6
    returns = rng.normal(0.0003, 0.012, size=(n_obs, n_assets))
    close = 100.0 * np.cumprod(1.0 + returns, axis=0)
    volume = rng.integers(100_000, 2_000_000, size=(n_obs, n_assets)).astype(float)

    momentum = rflux.factorlab.momentum(close, 20)
    forward = np.vstack([close[1:] / close[:-1] - 1.0, np.full((1, n_assets), np.nan)])
    ic = rflux.factorlab.information_coefficient(momentum, forward)
    print(f"momentum IC mean: {np.nanmean(ic):.4f}")

    market = close.mean(axis=1)
    market_returns = market[1:] / market[:-1] - 1.0
    model = rflux.fit_hmm(market_returns, n_states=2, seed=7, n_init=2)
    try:
        sticky = rflux.regimelab.sticky_regime(
            model, market_returns, min_duration=10
        )
        print(f"sticky states: {np.bincount(sticky.labels).tolist()}")
    finally:
        model.close()

    covariance = np.cov(returns, rowvar=False)
    weights = rflux.portfoliolab.hrp_weights(covariance)
    print(f"HRP weights: {np.round(weights, 4).tolist()}")

    features = rng.normal(size=(120, 3))
    target = 0.5 + 1.2 * features[:, 0] - 0.7 * features[:, 1]
    target[0] += 30.0
    fit = rflux.mllab.huber_regression(features, target)
    print(f"huber coefficients: {np.round(fit.coefficients, 4).tolist()}")

    plan = rflux.execution.almgren_chriss(
        10_000,
        slices=8,
        horizon=1.0,
        temporary_impact=0.02,
        permanent_impact=0.001,
        volatility=0.02,
        risk_aversion=1.5,
    )
    print(f"execution first slice: {plan.trades[0]:.1f}, cost: {plan.expected_cost:.2f}")

    tails = rflux.risk.tail_dependence(returns, q=0.05)
    print(f"tail dependence (0, 1): {tails[0, 1]:.4f}")


if __name__ == "__main__":
    main()
