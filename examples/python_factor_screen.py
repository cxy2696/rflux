#!/usr/bin/env python3
"""Cross-sectional factor screening with sector neutralization.

    PYTHONPATH=python python examples/python_factor_screen.py

The example uses a synthetic multi-asset panel so it runs without network access.
The same steps apply to a real wide price table from :mod:`rflux.data`:

1. build point-in-time factors at each date;
2. winsorize/z-score them across the universe;
3. neutralize each factor by sector;
4. combine the neutral factors;
5. align the factor at t with the return from t to t+1;
6. report IC, quantile spreads and a cost-aware long-short backtest.
"""

from __future__ import annotations

import argparse

import numpy as np

from rflux import backtest, qflux


def simulate_prices(
    seed: int, days: int, assets: int, sectors: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return prices with shape (days + 1, assets) and sector ids."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0002, 0.009, days)
    sector_shocks = rng.normal(0.0, 0.004, (days, sectors))
    sector_ids = np.arange(assets) % sectors
    betas = np.linspace(0.65, 1.2, assets)
    idio_vol = np.linspace(0.006, 0.014, assets)

    idio = rng.normal(0.0, 1.0, (days, assets)) * idio_vol[None, :]
    returns = (
        market[:, None] * betas[None, :]
        + sector_shocks[:, sector_ids]
        + idio
    )
    prices = np.vstack(
        [
            np.full(assets, 100.0),
            100.0 * np.cumprod(1.0 + returns, axis=0),
        ]
    )
    return prices, np.asarray(sector_ids, dtype=np.int32)


def cross_sectional_score(raw: np.ndarray, sectors: np.ndarray) -> np.ndarray:
    """Winsorize, z-score and sector-neutralize one cross-section."""
    clipped = qflux.winsorize(raw, 0.05, 0.95)
    standardized = qflux.zscore(clipped)
    return qflux.neutralize(standardized, sectors)


def build_factors(
    prices: np.ndarray, sectors: np.ndarray, lookback: int
) -> tuple[np.ndarray, np.ndarray]:
    """Build point-in-time factor rows and next-period returns.

    The returned arrays have shape (periods, assets).  Factor row ``i`` uses
    prices through date ``t`` and is matched with the return from ``t`` to
    ``t + 1``; no future price is used in the factor calculation.
    """
    if prices.shape[0] <= lookback + 1:
        raise ValueError("not enough price observations for the requested lookback")
    if lookback < 5:
        raise ValueError("lookback must be at least 5")

    periods = prices.shape[0] - 1 - lookback
    assets = prices.shape[1]
    factor_panel = np.full((periods, assets), np.nan)
    forward_panel = np.full((periods, assets), np.nan)

    for row, t in enumerate(range(lookback, prices.shape[0] - 1)):
        momentum = prices[t] / prices[t - lookback] - 1.0
        reversal = -(prices[t] / prices[t - 5] - 1.0)
        window = prices[t - lookback + 1:t + 1] / prices[t - lookback:t] - 1.0
        low_volatility = -np.std(window, axis=0, ddof=1)

        components = np.column_stack([momentum, low_volatility, reversal])
        neutral = np.column_stack(
            [
                cross_sectional_score(components[:, feature], sectors)
                for feature in range(components.shape[1])
            ]
        )
        factor_panel[row] = np.nanmean(neutral, axis=1)
        forward_panel[row] = prices[t + 1] / prices[t] - 1.0

    return factor_panel, forward_panel


def format_percent(value: float) -> str:
    return "nan" if not np.isfinite(value) else f"{value * 100:+.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--days", type=int, default=720)
    parser.add_argument("--assets", type=int, default=18)
    parser.add_argument("--sectors", type=int, default=4)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--quantiles", type=int, default=3)
    parser.add_argument("--cost-bps", type=float, default=5.0)
    args = parser.parse_args()
    if args.assets < 1 or args.sectors < 1:
        parser.error("assets and sectors must be positive")
    if args.lookback < 5:
        parser.error("lookback must be at least 5")
    if args.days <= args.lookback + 1:
        parser.error("days must be greater than lookback + 1")
    if args.quantiles < 2:
        parser.error("quantiles must be at least 2")
    if args.cost_bps < 0:
        parser.error("cost-bps must be non-negative")

    prices, sectors = simulate_prices(
        args.seed, args.days, args.assets, args.sectors
    )
    factor_panel, forward_panel = build_factors(
        prices, sectors, args.lookback
    )

    ic = backtest.factor_ic(factor_panel, forward_panel, method="spearman")
    ic_summary = backtest.summarize_ic(ic)
    quantile_returns = backtest.quantile_portfolio_returns(
        factor_panel, forward_panel, args.quantiles
    )
    result = backtest.long_short_backtest(
        factor_panel,
        forward_panel,
        n_quantiles=args.quantiles,
        cost_bps=args.cost_bps,
    )

    print(
        f"universe: {args.days} days, {args.assets} assets, "
        f"{args.sectors} sectors"
    )
    print(
        f"factor rows: {factor_panel.shape[0]}, "
        f"IC periods: {ic_summary.n_periods}"
    )
    print(
        f"rank IC: mean={ic_summary.mean:+.4f}, "
        f"ICIR(per period)={ic_summary.information_ratio:+.3f}, "
        f"positive={ic_summary.positive_ratio:.1%}"
    )
    quantile_mean = np.nanmean(quantile_returns, axis=1) * 10000.0
    print(
        "mean one-day forward return by factor quantile (bps):",
        np.round(quantile_mean, 2).tolist(),
    )

    metrics = result.metrics
    if metrics is not None:
        print(
            "long-short: "
            f"total={format_percent(metrics.total_return)}, "
            f"ann.return={format_percent(metrics.annualized_return)}, "
            f"ann.vol={metrics.annualized_volatility * 100:.2f}%, "
            f"Sharpe={metrics.sharpe:.2f}, "
            f"maxDD={format_percent(metrics.max_drawdown)}, "
            f"turnover={np.nanmean(result.turnover):.2f}"
        )

    latest = factor_panel[-1]
    ranked = np.argsort(latest)[::-1]
    positive = [int(index) for index in ranked if np.isfinite(latest[index])]
    print("latest top 5 factor scores:", positive[:5])
    print(
        f"cost assumption: {args.cost_bps:.1f} bps per one-way turnover; "
        "synthetic data, not an investment result"
    )


if __name__ == "__main__":
    main()
