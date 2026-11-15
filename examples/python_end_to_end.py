#!/usr/bin/env python3
"""End-to-end synthetic research demo.

Run from the repository root:

    PYTHONPATH=python RFLUX_LIBRARY=$PWD/build/librflux.dylib \
        python examples/python_end_to_end.py

The pipeline is deliberately explicit about time alignment:

1. simulate a two-regime multi-asset return panel;
2. build 20/60-day factor features without look-ahead;
3. fit a Gaussian HMM to the market return and inspect causal filtered states;
4. build a correlation network from the most recent 120 observations;
5. run bond percolation and an SIR risk-diffusion scenario;
6. evaluate a simple 60-day momentum long-short portfolio with forward returns.
"""

from __future__ import annotations

import argparse

import numpy as np

from rflux import backtest, metrics, percolab, qflux, regimelab


def simulate(seed: int = 2024, days: int = 720, assets: int = 6):
    rng = np.random.default_rng(seed)
    market = np.empty(days)
    for t in range(days):
        if t < int(days * 0.40):
            drift, vol = 0.0008, 0.008   # calm up-trend
        elif t < int(days * 0.72):
            drift, vol = 0.0000, 0.018   # choppy, higher volatility
        else:
            drift, vol = -0.0012, 0.030  # stress
        market[t] = rng.normal(drift, vol)
    betas = np.linspace(0.6, 1.3, assets)
    panel = (
        market[:, None] * betas[None, :]
        + rng.normal(0.0, 0.004, size=(days, assets))
        + rng.normal(0.0, 0.0002, size=(days, assets))
    )
    return panel, market


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--days", type=int, default=720)
    parser.add_argument("--assets", type=int, default=6)
    args = parser.parse_args()

    panel, market = simulate(args.seed, args.days, args.assets)
    market_prices = np.cumprod(1.0 + market)
    print("simulated panel:", panel.shape)
    print(
        "market summary:",
        metrics.performance_summary(market[1:], periods_per_year=252),
    )

    # Point-in-time factors: both inputs use data at or before t.
    mom20 = qflux.rolling_mean(market_prices, 20)
    mom60 = qflux.rolling_mean(market_prices, 60)
    trend = qflux.rolling_slope(market_prices, 60)
    r2 = qflux.rolling_r2(market_prices, 60)
    print(
        "last market features: mom20=%.3f mom60=%.3f slope60=%.5f r2=%.3f"
        % (mom20[-1], mom60[-1], trend[-1], r2[-1])
    )

    regime_model = regimelab.fit_hmm(
        market.reshape(-1, 1),
        n_states=3,
        max_iter=250,
        n_init=4,
        seed=args.seed,
    )
    try:
        filtered = regime_model.filtered(market)
        labels = np.argmax(filtered, axis=1)
        print("HMM fit:", regime_model.fit_info)
        print("HMM state counts:", np.bincount(labels, minlength=3).tolist())
        print("last filtered probabilities:", np.round(filtered[-1], 4))
        print("transition matrix:\n", np.round(regime_model.parameters.transition, 4))
        forecast_5d = regime_model.forecast(filtered[-1], horizon=5)
        print("5-day state forecast:", np.round(forecast_5d, 4))
    finally:
        regime_model.close()

    volatility = regimelab.volatility_regime(market, 20, 120)
    trend_regime = regimelab.trend_regime(market_prices, 20, 60)
    composite = regimelab.composite_regime(
        volatility.labels, trend_regime.labels
    )
    print(
        "causal regime counts: volatility=%s composite=%s"
        % (
            np.bincount(volatility.labels[volatility.labels >= 0], minlength=3).tolist(),
            np.bincount(composite[composite >= 0], minlength=3).tolist(),
        )
    )

    window = panel[-120:]
    network = percolab.correlation_network(window, threshold=0.45)
    print(
        "network: edges=%d density=%.3f components=%d largest=%d avg_clustering=%.3f"
        % (
            network.n_edges,
            network.density,
            network.components.n_components,
            network.components.largest_size,
            network.average_clustering,
        )
    )
    print("eigenvector centrality:", np.round(network.eigenvector, 4))

    scan = percolab.bond_percolation(
        network.adjacency,
        np.array([0.0, 0.25, 0.5, 0.75, 1.0]),
        repetitions=300,
        seed=args.seed,
    )
    print("percolation giant mean:", np.round(scan.giant_mean, 4))
    diffusion = percolab.risk_diffusion(
        network.adjacency,
        [int(np.argmax(network.eigenvector))],
        beta=2.0,
        gamma=0.15,
        steps=30,
        repetitions=300,
        seed=args.seed,
    )
    print(
        "risk diffusion: day0=%.3f day10=%.3f final=%.3f"
        % (
            diffusion.mean_affected[0],
            diffusion.mean_affected[10],
            diffusion.mean_affected[-1],
        )
    )

    # Six-asset cross-section is small, but the mechanics generalise.  Factor
    # and forward return are aligned so factor[t] is known before return[t].
    prices = np.cumprod(1.0 + panel, axis=0)
    factor = np.column_stack(
        [qflux.rolling_mean(prices[:, j], 60) for j in range(panel.shape[1])]
    )
    forward = backtest.forward_returns(prices, periods=1)
    valid_rows = np.isfinite(factor).all(axis=1) & np.isfinite(forward).all(axis=1)
    factor = factor[valid_rows]
    forward = forward[valid_rows]
    ic = backtest.factor_ic(factor, forward)
    print("factor IC summary:", backtest.summarize_ic(ic))
    backtest_result = backtest.long_short_backtest(
        factor,
        forward,
        n_quantiles=3,
        cost_bps=5.0,
        periods_per_year=252.0,
    )
    print("long-short metrics:", backtest_result.metrics)
    print("final equity:", round(float(backtest_result.equity[-1]), 4))


if __name__ == "__main__":
    main()
