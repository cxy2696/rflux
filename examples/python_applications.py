#!/usr/bin/env python3
"""Run the portfolio-risk and regime-attribution applications on sample data."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from rflux import apps, qflux, regimelab
from rflux.cli import read_price_csv


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    sample = root / "examples/data/sample_prices.csv"
    parsed = read_price_csv(sample)
    returns = np.column_stack(
        [qflux.returns(parsed.prices[:, column]) for column in range(parsed.prices.shape[1])]
    )
    weights = np.array([0.30, 0.25, 0.20, 0.15, 0.10])

    risk = apps.portfolio_risk(returns, weights, names=parsed.names)
    print("portfolio risk")
    print(f"  annual volatility: {risk['portfolio']['annualized_volatility']:.4%}")
    print(f"  diversification ratio: {risk['portfolio']['diversification_ratio']:.3f}")
    print(f"  gross / net exposure: {risk['input']['gross_exposure']:.2f} / {risk['input']['net_exposure']:.2f}")
    for name, item in risk["asset_risk"].items():
        contribution = item["component_volatility"]
        share = item["percent_contribution"]
        print(
            f"  {name}: weight={item['weight']:.2f}, "
            f"contribution={contribution:+.2%}, share={share:+.1%}"
        )

    valid = np.isfinite(returns)
    counts = valid.sum(axis=1)
    market = np.full(returns.shape[0], np.nan)
    mask = counts > 0
    market[mask] = np.nansum(returns, axis=1)[mask] / counts[mask]
    regime = regimelab.volatility_regime(market)
    attribution = apps.regime_attribution(market, regime.labels)
    print("\nregime attribution")
    for state, item in attribution["states"].items():
        annual_vol = item["annualized_volatility"]
        annual_vol_text = "n/a" if annual_vol is None else f"{annual_vol:.2%}"
        print(
            f"  state {state}: periods={item['periods']}, "
            f"mean={item['mean_return']:+.4%}, "
            f"annual vol={annual_vol_text}, "
            f"share of arithmetic return={item['share_of_total_sum']:+.1%}"
        )
    print("transition counts:")
    for row in attribution["transitions"]["counts"]:
        print("  ", row)


if __name__ == "__main__":
    main()
