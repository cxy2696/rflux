#!/usr/bin/env python3
"""Example with akshare, yfinance, tushare or a custom CSV.

Install the optional dependencies, for example:

    pip install 'rflux[yfinance]'
    PYTHONPATH=python python examples/python_real_data.py \
        --source yfinance --symbols AAPL MSFT NVDA --start 2021-01-01

    pip install 'rflux[tushare]'
    PYTHONPATH=python python examples/python_real_data.py \
        --source tushare --symbols 000001.SZ 600519.SH \
        --start 2021-01-01 --token "$TUSHARE_TOKEN"

    PYTHONPATH=python python examples/python_real_data.py \
        --source csv --path my_prices.csv --date-column date

This script downloads data only when explicitly run.  It is not part of the
test suite and makes no claim about the profitability of any signal.
"""

from __future__ import annotations

import argparse

import numpy as np

from rflux import data, metrics, percolab, qflux, regimelab


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", choices=["akshare", "yfinance", "tushare", "csv"],
        required=True,
    )
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--path", default=None, help="CSV path for --source csv")
    parser.add_argument("--date-column", default="date")
    parser.add_argument("--token", default=None, help="tushare token")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--threshold", type=float, default=0.45)
    args = parser.parse_args()

    if args.source == "csv":
        if not args.path:
            parser.error("--path is required with --source csv")
        prices = data.load_csv(args.path, date_column=args.date_column)
    else:
        if not args.symbols:
            parser.error("--symbols is required for provider sources")
        if args.source == "akshare":
            prices = data.load_akshare(
                args.symbols, start=args.start, end=args.end
            )
        elif args.source == "yfinance":
            prices = data.load_yfinance(
                args.symbols, start=args.start, end=args.end
            )
        else:
            prices = data.load_tushare(
                args.symbols, start=args.start, end=args.end,
                token=args.token,
            )
    returns = data.to_returns(prices, method="simple", fill_method="ffill")
    print("price panel:", prices.shape, prices.index.min(), prices.index.max())

    market = returns.mean(axis=1).dropna()
    print("market performance:", metrics.performance_summary(market.to_numpy()))

    for column in returns.columns:
        series = returns[column].dropna()
        if len(series) < 80:
            continue
        print(
            f"{column}: ann.vol={series.std() * np.sqrt(252):.3f} "
            f"RSI={qflux.rsi(series.to_numpy(), 14)[-1]:.1f} "
            f"Sharpe={qflux.sharpe(series.to_numpy(), 252, 0):.2f}"
        )

    recent = returns.dropna(how="all").tail(252).to_numpy()
    network = percolab.correlation_network(recent, threshold=args.threshold)
    print(
        "network edges=%d components=%d largest=%d avg_clustering=%.3f"
        % (
            network.n_edges,
            network.components.n_components,
            network.components.largest_size,
            network.average_clustering,
        )
    )
    print("centrality:", dict(zip(returns.columns, np.round(network.eigenvector, 4))))

    model = regimelab.fit_hmm(market.to_numpy().reshape(-1, 1), n_states=3, seed=42)
    try:
        filtered = model.filtered(market.to_numpy())
        print("fit:", model.fit_info)
        print("last filtered regime probabilities:", np.round(filtered[-1], 4))
    finally:
        model.close()


if __name__ == "__main__":
    main()
