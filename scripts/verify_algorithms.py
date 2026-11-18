#!/usr/bin/env python3
"""Deterministic invariant checks for every public rflux algorithm family.

This driver complements the unit tests: it calls each public family through the
same Python entry points applications use, checks cross-kernel invariants, and
writes a timestamped JSON trace under ``artifacts/validation/``.  Run it once
with the native extension and once with ``RFLUX_DISABLE_NATIVE=1`` to verify
both drivers.

It is deliberately dependency-light: NumPy and rflux only, no pandas/scipy.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import rflux  # noqa: E402
from rflux import (  # noqa: E402
    advice,
    analysis,
    apps,
    backtest,
    execution,
    factorlab,
    metrics,
    mllab,
    percolab,
    portfoliolab,
    qflux,
    regimelab,
    risk,
)



def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _warmup_mask(array: np.ndarray, count: int) -> np.ndarray:
    return np.isnan(np.asarray(array, dtype=np.float64)[:count])


class Runner:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.failed = 0

    def check(self, name: str, function: Callable[[], str]) -> None:
        start = time.perf_counter()
        try:
            detail = function()
            status = "PASS"
        except Exception as exc:  # noqa: BLE001 - report every algorithm failure
            status = "FAIL"
            detail = f"{type(exc).__name__}: {exc}"
            self.failed += 1
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self.rows.append(
            {
                "name": name,
                "status": status,
                "elapsed_ms": round(elapsed_ms, 3),
                "detail": detail,
            }
        )
        print(f"{status:4}  {elapsed_ms:9.2f} ms  {name:34}  {detail}")

    def benchmark(
        self, name: str, function: Callable[[], Any], repeats: int = 3
    ) -> None:
        best = float("inf")
        for _ in range(repeats):
            start = time.perf_counter()
            function()
            best = min(best, time.perf_counter() - start)
        self.rows.append(
            {
                "name": f"benchmark::{name}",
                "status": "INFO",
                "elapsed_ms": round(best * 1000.0, 3),
                "detail": f"best of {repeats}",
            }
        )
        print(f"INFO  {best * 1000.0:9.2f} ms  benchmark::{name}")


def _make_data() -> dict[str, Any]:
    rng = np.random.default_rng(20261118)
    n_obs, n_assets = 360, 5
    returns = rng.normal(0.0004, 0.01, size=(n_obs, n_assets))
    returns[:, 4] = 0.4 * returns[:, 0] + rng.normal(0.0, 0.004, n_obs)
    prices = 100.0 * np.cumprod(1.0 + returns, axis=0)
    volume = rng.integers(1_000, 10_000, size=prices.shape).astype(np.float64)
    series = prices[:, 0].copy()
    high = series * (1.0 + np.abs(rng.normal(0.0, 0.004, n_obs)))
    low = series * (1.0 - np.abs(rng.normal(0.0, 0.004, n_obs)))
    return {
        "rng": rng,
        "prices": prices,
        "returns": qflux.returns_panel(prices),
        "volume": volume,
        "series": series,
        "high": high,
        "low": low,
    }


def _make_graph(seed: int = 7, n: int = 36) -> np.ndarray:
    rng = np.random.default_rng(seed)
    adjacency = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        j = (i + 1) % n
        adjacency[i, j] = adjacency[j, i] = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < 0.12:
                weight = float(rng.integers(1, 4))
                adjacency[i, j] = adjacency[j, i] = weight
    return adjacency


def check_qflux_returns(data: dict[str, Any]) -> str:
    prices = data["prices"]
    returns = qflux.returns(prices[:, 0])
    panel = qflux.returns_panel(prices)
    _require(returns.shape == prices[:, 0].shape, "returns shape")
    _require(np.isnan(returns[0]), "returns warm-up")
    _require(panel.shape == prices.shape, "panel shape")
    expected = prices[1:] / prices[:-1] - 1.0
    np.testing.assert_allclose(panel[1:], expected, rtol=1e-12, atol=1e-14)
    return f"panel={panel.shape}, first-row NaN"


def check_qflux_windows(data: dict[str, Any]) -> str:
    values = data["series"]
    window = 20
    outputs = {
        "sma": qflux.sma(values, window),
        "mean": qflux.rolling_mean(values, window),
        "sum": qflux.rolling_sum(values, window),
        "std": qflux.rolling_std(values, window),
        "min": qflux.rolling_min(values, window),
        "max": qflux.rolling_max(values, window),
        "zscore": qflux.rolling_zscore(values, window),
        "quantile": qflux.rolling_quantile(values, window, 0.4),
        "median": qflux.rolling_median(values, window),
    }
    for name, output in outputs.items():
        _require(output.shape == values.shape, f"{name} shape")
        _require(_warmup_mask(output, window - 1).all(), f"{name} warm-up")
        _require(np.isfinite(output[-1]), f"{name} last value")
    last = values[-window:]
    np.testing.assert_allclose(outputs["mean"][-1], np.mean(last), rtol=1e-12)
    np.testing.assert_allclose(outputs["sum"][-1], np.sum(last), rtol=1e-12)
    np.testing.assert_allclose(outputs["std"][-1], np.std(last), rtol=1e-12)
    np.testing.assert_allclose(outputs["min"][-1], np.min(last), rtol=1e-12)
    np.testing.assert_allclose(outputs["max"][-1], np.max(last), rtol=1e-12)
    np.testing.assert_allclose(
        outputs["quantile"][-1], np.quantile(last, 0.4), rtol=1e-12
    )
    np.testing.assert_allclose(
        outputs["median"][-1], np.median(last), rtol=1e-12
    )
    np.testing.assert_allclose(outputs["sma"], outputs["mean"], equal_nan=True)
    return f"{len(outputs)} kernels, last-window reference matched"


def check_qflux_pairs(data: dict[str, Any]) -> str:
    x = data["series"]
    y = 0.7 * x + 0.3 * data["prices"][:, 1]
    window = 24
    corr = qflux.rolling_corr(x, y, window)
    cov = qflux.rolling_cov(x, y, window)
    beta = qflux.rolling_beta(x, y, window)
    slope = qflux.rolling_slope(x, window)
    r2 = qflux.rolling_r2(x, window)
    for name, output in {
        "corr": corr, "cov": cov, "beta": beta, "slope": slope, "r2": r2
    }.items():
        _require(output.shape == x.shape, f"{name} shape")
        _require(_warmup_mask(output, window - 1).all(), f"{name} warm-up")
        _require(np.isfinite(output[-1]), f"{name} last value")
    np.testing.assert_allclose(
        corr[-1], np.corrcoef(x[-window:], y[-window:])[0, 1], rtol=1e-10
    )
    np.testing.assert_allclose(
        cov[-1], np.cov(x[-window:], y[-window:], ddof=0)[0, 1], rtol=1e-10
    )
    expected_beta = np.cov(
        x[-window:], y[-window:], ddof=0
    )[0, 1] / np.var(y[-window:])
    np.testing.assert_allclose(beta[-1], expected_beta, rtol=1e-10)
    expected_slope = np.polyfit(np.arange(window), x[-window:], 1)[0]
    np.testing.assert_allclose(slope[-1], expected_slope, rtol=1e-10)
    _require(0.0 <= r2[-1] <= 1.0 + 1e-12, "rolling R2 range")
    return "corr/cov/beta/slope/R2 matched last-window references"


def check_qflux_indicators(data: dict[str, Any]) -> str:
    values = data["series"]
    high, low = data["high"], data["low"]
    ema = qflux.ema(values, 20)
    rsi = qflux.rsi(values, 14)
    atr = qflux.atr(high, low, values, 14)
    bands = qflux.bollinger(values, 20, 2.0)
    macd = qflux.macd(values)
    drawdown = qflux.drawdown(values)
    _require(np.isfinite(ema[-1]) and np.isfinite(rsi[-1]), "EMA/RSI last")
    _require(np.nanmin(rsi) >= 0.0 and np.nanmax(rsi) <= 100.0, "RSI range")
    _require(np.nanmin(atr) >= 0.0, "ATR non-negative")
    finite_bands = (
        np.isfinite(bands.upper)
        & np.isfinite(bands.middle)
        & np.isfinite(bands.lower)
    )
    _require(
        np.all(
            (bands.upper[finite_bands] >= bands.middle[finite_bands] - 1e-12)
            & (bands.middle[finite_bands] >= bands.lower[finite_bands] - 1e-12)
        ),
        "Bollinger ordering",
    )
    _require(macd.macd.shape == values.shape, "MACD shape")
    _require(macd.signal.shape == values.shape, "MACD signal shape")
    _require(drawdown[0] == 0.0 and np.nanmax(drawdown) <= 0.0, "drawdown")
    return "EMA/RSI/ATR/Bollinger/MACD/drawdown shapes and ranges"


def check_qflux_stats(data: dict[str, Any]) -> str:
    values = data["series"]
    returns = data["returns"][:, 0]
    returns = returns[np.isfinite(returns)]
    moments = qflux.describe(values)
    np.testing.assert_allclose(moments.mean, np.mean(values), rtol=1e-12)
    np.testing.assert_allclose(moments.variance, np.var(values, ddof=1), rtol=1e-12)
    np.testing.assert_allclose(
        qflux.quantile(values, 0.35), np.quantile(values, 0.35), rtol=1e-12
    )
    np.testing.assert_allclose(
        qflux.pearson(values, data["prices"][:, 1]),
        np.corrcoef(values, data["prices"][:, 1])[0, 1],
        rtol=1e-10,
    )
    corr = qflux.correlation_matrix(data["returns"])
    complete = data["returns"][np.all(np.isfinite(data["returns"]), axis=1)]
    reference = np.corrcoef(complete, rowvar=False)
    np.testing.assert_allclose(corr, reference, rtol=1e-10, atol=1e-12)
    sharpe = qflux.sharpe(returns)
    sortino = qflux.sortino(returns)
    tail = qflux.var_cvar(returns, 0.95)
    _require(np.isfinite(sharpe) and np.isfinite(sortino), "performance ratios")
    _require(np.isfinite(tail.var) and np.isfinite(tail.cvar), "tail risk")
    _require(tail.var >= 0.0 and tail.cvar >= tail.var - 1e-12, "tail ordering")
    return "describe/quantile/Pearson/correlation/Sharpe/Sortino/VaR"


def check_qflux_ranking(data: dict[str, Any]) -> str:
    values = data["series"].copy()
    values[::37] = np.nan
    ranks = qflux.rank(values)
    percentiles = qflux.percentile_rank(values)
    zscore = qflux.zscore(values)
    winsor = qflux.winsorize(values, 0.05, 0.95)
    labels = np.arange(values.size) % 4
    labels[::11] = -1
    neutral = qflux.neutralize(values, labels, 4)
    valid = np.isfinite(values)
    _require(np.all(np.isnan(ranks[~valid])), "rank NaN passthrough")
    _require(np.nanmin(ranks[valid]) >= 0.0, "rank non-negative")
    _require(
        np.all(np.isnan(percentiles[~valid]))
        and np.nanmax(percentiles[valid]) <= 1.0,
        "percentile range",
    )
    _require(np.nanmax(np.abs(zscore[valid])) > 0.0, "zscore non-trivial")
    _require(
        np.nanmin(winsor[valid]) >= np.nanquantile(values[valid], 0.05) - 1e-12,
        "winsor lower",
    )
    _require(
        np.nanmax(winsor[valid]) <= np.nanquantile(values[valid], 0.95) + 1e-12,
        "winsor upper",
    )
    _require(np.all(np.isnan(neutral[labels < 0])), "neutralize mask")
    return "rank/percentile/zscore/winsorize/neutralize"


def check_factors(data: dict[str, Any]) -> str:
    prices, volume, returns = data["prices"], data["volume"], data["returns"]
    momentum = factorlab.momentum(prices, 20)
    reversal = factorlab.volume_reversal(prices, volume, 20)
    vwap = factorlab.vwap_deviation(prices, volume, 20)
    amihud = factorlab.amihud(prices, volume, 20)
    for name, factor, warmup in (
        ("momentum", momentum, 20),
        ("volume_reversal", reversal, 20),
        ("vwap", vwap, 19),
        ("amihud", amihud, 20),
    ):
        _require(factor.shape == prices.shape, f"{name} shape")
        _require(np.all(np.isnan(factor[:warmup])), f"{name} warm-up")
        _require(np.any(np.isfinite(factor[warmup:])), f"{name} finite tail")
    _require(np.nanmin(amihud[20:]) >= 0.0, "Amihud non-negative")
    ic = factorlab.information_coefficient(momentum, returns)
    turnover = factorlab.factor_turnover(np.nan_to_num(np.abs(momentum)))
    _require(ic.shape == (prices.shape[0],), "IC shape")
    _require(turnover.shape == (prices.shape[0],), "turnover shape")
    _require(turnover[0] == 0.0, "first turnover zero")
    return "momentum/volume-reversal/VWAP/Amihud/IC/turnover"


def check_hmm(data: dict[str, Any]) -> str:
    rng = data["rng"]
    observations = np.concatenate(
        [
            rng.normal(-0.008, 0.004, 220),
            rng.normal(0.012, 0.006, 220),
        ]
    ).reshape(-1, 1)
    model = regimelab.fit_hmm(
        observations, 2, max_iter=120, n_init=2, seed=17
    )
    try:
        fit = model.fit_info
        params = model.parameters
        filtered = model.filtered(observations)
        smoothed = model.smoothed(observations)
        viterbi = model.viterbi(observations)
        states = model.filtered_states(observations)
        forecast = model.forecast(filtered[-1], horizon=3)
        step = model.filter_step(filtered[-1], observations[-1])
        stationary = model.stationary_distribution()
        durations = model.expected_durations()
        _require(np.isfinite(fit.log_likelihood), "finite log likelihood")
        np.testing.assert_allclose(filtered.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(smoothed.sum(axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(forecast.sum(), 1.0, atol=1e-9)
        np.testing.assert_allclose(step.sum(), 1.0, atol=1e-9)
        np.testing.assert_allclose(stationary.sum(), 1.0, atol=1e-9)
        np.testing.assert_allclose(
            params.transition.sum(axis=1), 1.0, atol=1e-9
        )
        _require(viterbi.shape == (observations.shape[0],), "Viterbi shape")
        _require(np.all(np.isin(viterbi, np.arange(2))), "Viterbi state ids")
        _require(states.shape == (observations.shape[0],), "state shape")
        _require(np.all(durations > 0.0), "durations positive")
        _require(np.isfinite(model.score(observations)), "score finite")
        _require(
            np.isfinite(model.conditional_mean(filtered[-1])),
            "conditional mean",
        )
        np.testing.assert_array_equal(states, np.argmax(filtered, axis=1))
    finally:
        model.close()
    snapshot = regimelab.fit_hmm_snapshot(
        observations, 2, max_iter=120, n_init=2, seed=17, horizon=3
    )
    _require(snapshot.latest_filtered.shape == (2,), "snapshot filtered")
    _require(snapshot.forecast.shape == (2,), "snapshot forecast")
    np.testing.assert_allclose(snapshot.latest_filtered.sum(), 1.0, atol=1e-9)
    np.testing.assert_allclose(snapshot.forecast.sum(), 1.0, atol=1e-9)
    return f"k=2 fit/iter={fit.iterations}/filtered/smoothed/Viterbi/forecast"


def check_hmm_alignment(data: dict[str, Any]) -> str:
    means = np.array([[-0.01], [0.01]])
    variances = np.array([[0.004 ** 2], [0.006 ** 2]])
    reference = regimelab.HMMParameters(
        np.array([0.5, 0.5]),
        np.array([[0.9, 0.1], [0.2, 0.8]]),
        means,
        variances,
    )
    target = regimelab.HMMParameters(
        np.array([0.5, 0.5]),
        reference.transition[::-1, ::-1].copy(),
        means[::-1].copy(),
        variances[::-1].copy(),
    )
    order = regimelab.match_states(reference, target)
    _require(order.shape == (2,), "state order shape")
    aligned = regimelab.align_parameters(reference, target)
    _require(aligned.order.shape == (2,), "alignment order")
    np.testing.assert_allclose(
        aligned.parameters.means, reference.means, atol=1e-12
    )
    probabilities = np.array([[0.7, 0.3], [0.4, 0.6]])
    moved = regimelab.align_probabilities(probabilities, order)
    np.testing.assert_allclose(moved.sum(axis=1), 1.0, atol=1e-12)
    labels = np.array([0, 1, 0, -1], dtype=np.int32)
    moved_labels = regimelab.align_labels(labels, order)
    _require(moved_labels[-1] == -1, "aligned label mask")
    return "Hungarian order/unbiased parameter and label alignment"


def check_regimes(data: dict[str, Any]) -> str:
    returns = data["returns"][:, 0]
    prices = data["prices"][:, 0]
    volatility = regimelab.volatility_regime(
        returns, vol_window=20, lookback=80
    )
    trend = regimelab.trend_regime(prices, 20, 60)
    composite = regimelab.composite_regime(volatility.labels, trend.labels)
    _require(
        volatility.labels.shape == returns.shape
        and trend.labels.shape == prices.shape,
        "regime shapes",
    )
    _require(
        np.all(volatility.labels[:20] == -1)
        and np.all(trend.labels[:59] == -1),
        "regime warm-up",
    )
    valid = (volatility.labels >= 0) & (trend.labels >= 0)
    expected = np.maximum(volatility.labels[valid], 2 - trend.labels[valid])
    np.testing.assert_array_equal(composite[valid], expected)
    _require(np.all(composite[~valid] == -1), "composite mask")
    observations = returns[np.isfinite(returns)].reshape(-1, 1)
    model = regimelab.fit_hmm(observations, 2, max_iter=60, n_init=1, seed=3)
    try:
        sticky = regimelab.sticky_regime(model, observations, min_duration=3)
        _require(
            sticky.labels.shape == (observations.shape[0],), "sticky shape"
        )
        _require(
            np.all(np.isin(sticky.labels, np.arange(2))), "sticky ids"
        )
    finally:
        model.close()
    return "volatility/trend/composite/sticky labels"


def check_network(data: dict[str, Any]) -> str:
    adjacency = _make_graph()
    n = adjacency.shape[0]
    strength = percolab.network_strength(adjacency)
    eigen = percolab.centrality(adjacency, "eigenvector")
    degree_centrality = percolab.centrality(adjacency, "degree")
    pagerank = percolab.pagerank(adjacency)
    between = percolab.betweenness(adjacency)
    clustering = percolab.clustering(adjacency)
    transitivity = percolab.transitivity(adjacency)
    paths = percolab.path_statistics(adjacency)
    cores = percolab.core_numbers(adjacency)
    assortativity = percolab.assortativity(adjacency)
    modularity = percolab.modularity(adjacency, np.arange(n) % 3)
    bridges = percolab.bridges(adjacency)
    articulation = percolab.articulation_points(adjacency)
    truss = percolab.truss_numbers(adjacency)
    for name, values in (
        ("strength", strength),
        ("eigenvector", eigen),
        ("degree", degree_centrality),
        ("pagerank", pagerank),
        ("betweenness", between),
        ("clustering", clustering),
        ("core", cores),
        ("articulation", articulation),
    ):
        _require(values.shape == (n,), f"{name} shape")
    _require(truss.shape == (n, n), "truss matrix shape")
    _require(bridges.shape[1] == 2 if bridges.size else True, "bridge pairs")
    np.testing.assert_allclose(pagerank.sum(), 1.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(eigen), 1.0, atol=1e-9)
    _require(np.nanmin(strength) >= 0.0, "strength non-negative")
    _require(np.nanmin(clustering) >= 0.0, "clustering lower")
    _require(np.nanmax(clustering) <= 1.0 + 1e-12, "clustering upper")
    _require(0.0 <= transitivity <= 1.0, "transitivity range")
    _require(paths.diameter >= 0 and paths.efficiency >= 0.0, "path stats")
    _require(np.isfinite(assortativity), "assortativity")
    _require(abs(modularity) <= 1.0 + 1e-12, "modularity range")
    return (
        f"n={n}, {len(bridges)} bridges, "
        f"{int(np.count_nonzero(articulation))} articulations"
    )


def check_network_mst(data: dict[str, Any]) -> str:
    corr = qflux.correlation_matrix(data["returns"])
    tree = percolab.mst_from_correlation(corr)
    distance = percolab.correlation_to_distance(corr)
    tree2 = percolab.mst_from_distance(distance)
    _require(tree.u.shape == (corr.shape[0] - 1,), "MST edge count")
    _require(tree2.weight.shape == (corr.shape[0] - 1,), "distance MST edges")
    np.testing.assert_allclose(
        np.sort(tree.weight), np.sort(tree2.weight), rtol=1e-12, atol=1e-14
    )
    clusters = percolab.mst_clusters(distance, 0.8)
    _require(clusters.n_components >= 1, "MST clusters")
    components = percolab.threshold_components(
        percolab.correlation_adjacency(corr, 0.1), 0.1
    )
    _require(components.largest_size >= 1, "threshold components")
    network = percolab.network_from_correlation(corr, 0.2)
    _require(network.adjacency.shape == corr.shape, "network adjacency")
    report = percolab.correlation_network(data["returns"], 0.2)
    _require(report.n_edges >= 0, "correlation network")
    return f"MST edges={tree.u.size}, clusters={clusters.n_components}"


def check_percolation(data: dict[str, Any]) -> str:
    adjacency = _make_graph()
    n = adjacency.shape[0]
    probabilities = [0.0, 0.25, 0.5, 0.75, 1.0]
    scan = percolab.percolation_scan(
        adjacency, probabilities, kind="bond", repetitions=40, seed=11
    )
    _require(scan.giant_mean.shape == (len(probabilities),), "scan shape")
    _require(scan.giant_mean[-1] == 1.0, "scan p=1 giant")
    _require(np.all(scan.giant_std >= 0.0), "scan std non-negative")
    _require(np.all(scan.susceptibility >= -1e-12), "scan susceptibility")
    repeat = percolab.percolation_scan(
        adjacency, probabilities, kind="bond", repetitions=40, seed=11
    )
    np.testing.assert_allclose(scan.giant_mean, repeat.giant_mean, atol=0.0)
    site = percolab.site_percolation(
        adjacency, probabilities, repetitions=30, seed=3
    )
    _require(site.giant_mean[0] == 0.0, "site p=0 giant")
    bond = percolab.bond_percolation(
        adjacency, probabilities, repetitions=30, seed=3
    )
    weighted_bond = percolab.bond_percolation_weighted(
        adjacency, probabilities, repetitions=30, seed=3
    )
    node_weights = np.linspace(0.2, 1.0, n)
    weighted_site = percolab.site_percolation_weighted(
        adjacency, node_weights, probabilities, repetitions=30, seed=3
    )
    for name, result in (
        ("site", site),
        ("bond", bond),
        ("weighted_bond", weighted_bond),
        ("weighted_site", weighted_site),
    ):
        _require(result.giant_mean.shape == (len(probabilities),), f"{name}")
        _require(
            np.all((result.giant_mean >= -1e-12) & (result.giant_mean <= 1.0 + 1e-12)),
            f"{name} fraction range",
        )
    sizes = percolab.cluster_size_distribution(
        adjacency, 0.5, kind="bond", repetitions=30, seed=3
    )
    _require(sizes.shape == (n + 1,), "cluster distribution shape")
    _require(sizes[0] == 0.0, "cluster size zero")
    return f"n={n}, p=0/1, site/bond/weighted + cluster distribution"


def check_removal_diffusion(data: dict[str, Any]) -> str:
    adjacency = _make_graph(seed=9)
    n = adjacency.shape[0]
    order = percolab.removal_order(adjacency, "strength")
    _require(np.array_equal(np.sort(order), np.arange(n)), "removal permutation")
    curve = percolab.targeted_removal(adjacency, order, checkpoints=6)
    _require(curve.removed.shape == (7,), "removal curve")
    _require(np.all(np.diff(curve.removed) >= 0), "removal monotone")
    core = percolab.core_percolation(adjacency)
    _require(core.k.shape == core.n_nodes.shape, "core scan shape")
    diffusion = percolab.risk_diffusion(
        adjacency, [0, 1], beta=0.3, gamma=0.1, steps=12, repetitions=30, seed=3
    )
    _require(diffusion.mean_affected.shape == (13,), "diffusion path")
    _require(
        diffusion.mean_affected[0] >= 2.0 / n - 1e-12,
        "diffusion initial seeds",
    )
    _require(
        np.all(diffusion.infection_frequency >= -1e-12)
        and np.all(diffusion.infection_frequency <= 1.0 + 1e-12),
        "infection frequency range",
    )
    return f"n={n}, removal/core/diffusion paths"


def check_portfolio(data: dict[str, Any]) -> str:
    complete = data["returns"][np.all(np.isfinite(data["returns"]), axis=1)]
    covariance = np.cov(complete, rowvar=False)
    hrp = portfoliolab.hrp_weights(covariance)
    nco = portfoliolab.nco_weights(covariance)
    parity = portfoliolab.risk_parity_weights(covariance)
    order = portfoliolab.quasi_diagonal_order(covariance)
    for name, weights in (("HRP", hrp), ("NCO", nco), ("parity", parity)):
        _require(weights.shape == (covariance.shape[0],), f"{name} shape")
        _require(np.all(weights >= -1e-15), f"{name} non-negative")
        np.testing.assert_allclose(weights.sum(), 1.0, atol=1e-10)
    _require(
        np.array_equal(np.sort(order), np.arange(covariance.shape[0])),
        "quasi-diagonal permutation",
    )
    current = np.full(covariance.shape[0], 1.0 / covariance.shape[0])
    target = hrp.copy()
    gated = portfoliolab.rebalance_targets(current, target, threshold=0.01)
    _require(gated.shape == current.shape, "rebalance shape")
    path = portfoliolab.rebalance_targets(
        np.vstack([current, current]), np.vstack([target, current]), threshold=0.01
    )
    _require(path.shape == (2, covariance.shape[0]), "rebalance path shape")
    return f"{len(hrp)} assets, HRP/NCO/parity/rebalance"


def check_mllab(data: dict[str, Any]) -> str:
    rng = data["rng"]
    features = rng.normal(size=(160, 2))
    target = 1.5 * features[:, 0] - 2.0 * features[:, 1] + 0.4
    fit = mllab.huber_regression(features, target, epsilon=1.35)
    np.testing.assert_allclose(fit.coefficients, [1.5, -2.0], atol=1e-8)
    np.testing.assert_allclose(fit.intercept, 0.4, atol=1e-8)
    predicted = fit.predict(features[:5])
    _require(predicted.shape == (5,), "Huber prediction shape")
    loss = mllab.cvar_loss(target, alpha=0.9)
    _require(np.isfinite(loss), "CVaR loss finite")
    adjacency = _make_graph(seed=2)
    signal = rng.normal(size=adjacency.shape[0])
    smoothed = mllab.graph_smooth(signal, adjacency, alpha=0.5, steps=4)
    _require(smoothed.shape == signal.shape, "graph smooth shape")
    _require(np.all(np.isfinite(smoothed)), "graph smooth finite")
    return "Huber IRLS/CVaR/graph smoothing"


def check_execution(data: dict[str, Any]) -> str:
    linear = execution.almgren_chriss(
        100.0, slices=5, horizon=1.0, temporary_impact=0.2
    )
    _require(linear.trades.shape == (5,), "linear trades shape")
    np.testing.assert_allclose(linear.trades.sum(), 100.0, atol=1e-9)
    np.testing.assert_allclose(linear.trades, np.full(5, 20.0), atol=1e-12)
    _require(linear.expected_cost > 0.0, "linear cost")
    curved = execution.almgren_chriss(
        100.0,
        slices=5,
        horizon=1.0,
        temporary_impact=0.2,
        volatility=0.3,
        risk_aversion=1.0,
    )
    np.testing.assert_allclose(curved.trades.sum(), 100.0, atol=1e-9)
    _require(np.all(curved.trades >= -1e-12), "curved trades")
    _require(curved.kappa > 0.0, "risk-averse kappa")
    book = execution.order_book_metrics(
        np.array([99.9, 100.0]),
        np.array([10.0, 20.0]),
        np.array([100.1, 100.2]),
        np.array([5.0, 15.0]),
    )
    _require(book["spread"].shape == (2,), "order-book shape")
    _require(np.all(book["mid"] > 0.0), "order-book mid")
    return "Almgren-Chriss linear/risk-averse + order-book metrics"


def check_risk(data: dict[str, Any]) -> str:
    values = data["returns"][:, :3].copy()
    values[::29, 1] = np.nan
    dependence = risk.tail_dependence(values, q=0.1)
    _require(dependence.shape == (3, 3), "tail dependence shape")
    np.testing.assert_allclose(dependence, dependence.T, equal_nan=True)
    np.testing.assert_allclose(np.diag(dependence), 1.0)
    upper = risk.tail_dependence(values, q=0.05, tail="upper")
    _require(upper.shape == (3, 3), "upper tail shape")
    return "lower/upper tail dependence symmetric"


def check_metrics(data: dict[str, Any]) -> str:
    returns = data["returns"][:, 0]
    returns = returns[np.isfinite(returns)]
    summary = metrics.performance_summary(returns)
    _require(np.isfinite(summary.sharpe), "performance Sharpe")
    _require(summary.n_periods == returns.size, "performance count")
    equity = metrics.equity_curve(returns)
    _require(equity.shape == (returns.size,), "equity shape")
    drawdown = metrics.drawdown_series(equity)
    _require(drawdown.shape == equity.shape, "drawdown shape")
    _require(np.nanmax(drawdown) <= 0.0, "drawdown sign")
    return f"n={returns.size}, Sharpe={summary.sharpe:.3f}"


def check_backtest(data: dict[str, Any]) -> str:
    prices = data["prices"]
    forward = backtest.forward_returns(prices, periods=1)
    _require(forward.shape == prices.shape, "forward shape")
    _require(np.all(np.isnan(forward[-1])), "forward tail NaN")
    factor = factorlab.momentum(prices, 20)
    ic = backtest.factor_ic(factor, forward)
    _require(ic.shape == (prices.shape[0],), "factor IC shape")
    quantiles = backtest.quantile_portfolio_returns(factor, forward, 4)
    _require(quantiles.shape == (4, prices.shape[0]), "quantile shape")
    result = backtest.long_short_backtest(
        factor, forward, n_quantiles=4, cost_bps=2.0
    )
    _require(result.net_returns.shape == (prices.shape[0],), "backtest shape")
    _require(result.equity.shape == (prices.shape[0] + 1,), "backtest equity")
    _require(np.all(result.costs >= 0.0), "backtest costs")
    stderr = backtest.summarize_ic(ic)
    _require(np.isfinite(stderr.mean), "IC summary mean")
    return f"IC mean={stderr.mean:.3f}, backtest turnover={result.turnover.sum():.3f}"


def check_apps(data: dict[str, Any]) -> str:
    returns = data["returns"]
    weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])
    report = apps.portfolio_risk(returns, weights)
    _require(len(report["input"]["asset_names"]) == 5, "portfolio risk assets")
    _require(
        report["input"]["gross_exposure"] > 0.0, "portfolio risk exposure"
    )
    labels = np.zeros(returns.shape[0], dtype=np.int32)
    labels[returns.shape[0] // 2 :] = 1
    regime = apps.regime_attribution(returns[:, 0], labels)
    _require(regime["input"]["states"] == 2, "regime attribution states")
    json.dumps(report, allow_nan=False)
    json.dumps(regime, allow_nan=False)
    return "portfolio risk + regime attribution JSON-safe"


def check_analysis(data: dict[str, Any]) -> str:
    report = analysis.analyze_prices(
        data["prices"],
        include_hmm=True,
        hmm_states=2,
        hmm_max_iter=60,
        hmm_n_init=1,
        include_network=True,
        network_threshold=0.2,
        seed=5,
    )
    _require(report["input"]["assets"] == 5, "analysis assets")
    _require(report["hmm"] is not None, "analysis HMM section")
    _require(report["network"] is not None, "analysis network section")
    json.dumps(report, allow_nan=False)
    return f"hmm states={report['hmm']['states']}, network edges={report['network']['edges']}"


def check_advice(data: dict[str, Any]) -> str:
    report = advice.trade_advice(
        data["prices"],
        include_hmm=True,
        hmm_states=2,
        hmm_max_iter=60,
        hmm_n_init=1,
        max_weight=0.6,
        target_volatility=0.15,
        portfolio_value=1_000_000.0,
        seed=5,
    )
    allocation = report["allocation"]
    target = np.asarray(allocation["target_weights"])
    _require(target.shape == (5,), "advice target shape")
    _require(np.all(target >= -1e-15), "advice non-negative")
    _require(target.sum() <= 1.0 + 1e-12, "advice no leverage")
    _require(
        abs(float(target.sum() + allocation["cash_weight"]) - 1.0) <= 1e-12,
        "advice cash identity",
    )
    _require(allocation["expected_turnover"] >= 0.0, "advice turnover")
    _require(report["regime"] is not None, "advice HMM section")
    json.dumps(report, allow_nan=False)
    return f"exposure={allocation['exposure']:.3f}, turnover={allocation['expected_turnover']:.3f}"


def _git_state() -> dict[str, str]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:  # noqa: BLE001 - diagnostics must not hide algorithm results
        commit = branch = "unknown"
    return {"commit": commit, "branch": branch}


def _benchmarks(runner: Runner, data: dict[str, Any]) -> None:
    rng = np.random.default_rng(99)
    values = rng.normal(size=250_000)
    runner.benchmark(
        "rolling_quantile n=250k w=256",
        lambda: qflux.rolling_quantile(values, 256, 0.5),
    )
    panel = rng.normal(size=(60, 4_000))
    runner.benchmark(
        "correlation_matrix 60x4000",
        lambda: qflux.correlation_matrix(panel),
    )
    adjacency = _make_graph(seed=4, n=220)
    runner.benchmark(
        "percolation_scan n=220 10p x 20rep",
        lambda: percolab.percolation_scan(
            adjacency, np.linspace(0.1, 0.9, 10), repetitions=20, seed=7
        ),
    )
    observations = rng.normal(size=(20_000, 1))
    runner.benchmark(
        "hmm_fit n=20k k=2 n_init=2",
        lambda: regimelab.fit_hmm(
            observations, 2, max_iter=40, n_init=2, seed=3
        ),
        repeats=2,
    )
    complete = data["returns"][np.all(np.isfinite(data["returns"]), axis=1)]
    covariance = np.cov(complete, rowvar=False)
    runner.benchmark("hrp 5 assets", lambda: portfoliolab.hrp_weights(covariance))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="", help="suffix for the report name")
    parser.add_argument(
        "--no-benchmark", action="store_true", help="skip the hot-kernel timings"
    )
    args = parser.parse_args()

    data = _make_data()
    runner = Runner()
    runner.check("qflux.returns", lambda: check_qflux_returns(data))
    runner.check("qflux.windows", lambda: check_qflux_windows(data))
    runner.check("qflux.pairs", lambda: check_qflux_pairs(data))
    runner.check("qflux.indicators", lambda: check_qflux_indicators(data))
    runner.check("qflux.stats", lambda: check_qflux_stats(data))
    runner.check("qflux.ranking", lambda: check_qflux_ranking(data))
    runner.check("factorlab.factors", lambda: check_factors(data))
    runner.check("regimelab.hmm", lambda: check_hmm(data))
    runner.check("regimelab.alignment", lambda: check_hmm_alignment(data))
    runner.check("regimelab.regimes", lambda: check_regimes(data))
    runner.check("percolab.network", lambda: check_network(data))
    runner.check("percolab.mst", lambda: check_network_mst(data))
    runner.check("percolab.percolation", lambda: check_percolation(data))
    runner.check("percolab.removal+diffusion", lambda: check_removal_diffusion(data))
    runner.check("portfoliolab.allocators", lambda: check_portfolio(data))
    runner.check("mllab.robust", lambda: check_mllab(data))
    runner.check("execution.microstructure", lambda: check_execution(data))
    runner.check("risk.tail", lambda: check_risk(data))
    runner.check("metrics.performance", lambda: check_metrics(data))
    runner.check("backtest.factor", lambda: check_backtest(data))
    runner.check("apps.reports", lambda: check_apps(data))
    runner.check("analysis.prices", lambda: check_analysis(data))
    runner.check("advice.rebalance", lambda: check_advice(data))
    if not args.no_benchmark:
        _benchmarks(runner, data)

    info = rflux.native_info()
    fallback = os.environ.get("RFLUX_DISABLE_NATIVE", "") == "1"
    backend = "ctypes-fallback" if fallback else info.get("module", "unknown")
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "label": args.label,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "rflux_version": rflux.__version__,
        "abi_version": rflux.abi_version(),
        "native": info,
        "backend": backend,
        "git": _git_state(),
        "checks": runner.rows,
        "failed": runner.failed,
    }
    output = ROOT / "artifacts" / "validation"
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    suffix = f"-{args.label}" if args.label else ""
    path = output / f"{stamp}-algorithms{suffix}.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    passed = sum(row["status"] == "PASS" for row in runner.rows)
    print()
    print(
        f"backend={backend} checks={passed} failed={runner.failed} "
        f"report={path.relative_to(ROOT)}"
    )
    return 0 if runner.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
