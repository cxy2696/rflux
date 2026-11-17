# Applications: portfolio risk, regime attribution and network risk

[中文](APPLICATIONS_cn.md) · [Back to English guide](../README_en.md)

These application-level functions sit on top of the tested `qflux`, `percolab` and `metrics` kernels. They add no hidden numerical layer; they make the input, estimator, annualisation and graph conventions explicit so a result can be checked later.

## 1. Portfolio risk

`rflux.portfolio_risk(returns, weights)` takes a panel of periodic arithmetic returns, with time on rows and assets on columns. Weights are exposures per unit of capital. They may be negative and do not need to sum to one.

### Sample and covariance

Keep rows where every asset has a finite return. The number of discarded rows is reported as `input.dropped_observations`. With the retained sample $R_c$ , the sample covariance is:

$$
\hat\Sigma = \frac{1}{N-d} R_c^\top R_c
$$

where $d$ is `ddof` (default 1). Using one complete-case sample, rather than pairwise-complete samples, keeps the covariance positive semidefinite and makes the portfolio variance well defined.

Portfolio return and variance:

$$
r_p = R_c w,\qquad
\sigma_p^2 = w^\top \hat\Sigma w
$$

Annualised volatility is $\sigma_p\sqrt{PY}$ , with `PY` defaulting to 252.

### Risk contributions

When portfolio volatility is positive, use the Euler decomposition:

$$
\mathrm{MCTR}_i=\frac{(\hat\Sigma w)_i}{\sigma_p},\qquad
\mathrm{CTR}_i=w_i\mathrm{MCTR}_i
$$

`component_volatility` is the annualised $\mathrm{CTR}_i\sqrt{PY}$ . The components sum to the portfolio's annualised volatility. `percent_contribution` is each component's share of that volatility; hedges can have a negative contribution.

The report also contains:

- `diversification_ratio = sum(|w_i| * asset_vol_i) / portfolio_vol`;
- `gross_exposure = sum(|w_i|)` and `net_exposure = sum(w_i)`;
- `gross_hhi = sum((w_i / gross_exposure)^2)`, a concentration measure of gross weights.

The remaining portfolio metrics come from `metrics.performance_summary`: total return, geometric annual return, volatility, Sharpe, Sortino, maximum drawdown and VaR/CVaR. Weights are not normalised, so `gross_hhi` describes the shape of the gross book, not net exposure. `ddof` affects the covariance and risk contributions only; the return metrics always use a sample standard deviation (`ddof=1`), recorded as `metrics_ddof`.

```python
import numpy as np
import rflux

returns = np.array([
    [0.01, 0.02],
    [0.02, -0.01],
    [-0.01, 0.01],
    [0.03, 0.00],
])
report = rflux.portfolio_risk(returns, [0.6, 0.4], names=["A", "B"])
print(report["portfolio"]["annualized_volatility"])
print(report["asset_risk"]["A"]["component_volatility"])
```

### Missing values and zero volatility

- Only rows where every asset is finite enter the covariance. This is not the pairwise-complete convention used by `correlation_matrix`.
- `ddof` must be smaller than the complete sample size.
- If a portfolio is perfectly hedged and volatility is zero, the asset contributions and diversification ratio are `None`; the return metrics are still computed.
- With more than two assets, weights, return columns and `names` must use the same order.

## 2. Regime attribution

`rflux.regime_attribution(returns, labels)` maps an existing state-label series to returns. Labels can come from HMM `filtered_states()` or from a rule-based regime. `-1` means undefined/warm-up and is excluded from state statistics and transitions.

For each state the report contains:

- `periods` and `period_share`: label count and share;
- `n_returns`: finite returns in that state;
- `mean_return` and `annualized_mean_return`: arithmetic mean and mean times `periods_per_year`, not geometric annualisation;
- `annualized_volatility`: sample standard deviation times $\sqrt{periods_per_year}$ ;
- `sharpe`, `var`, `cvar`, `hit_rate`;
- `sum_return` and `share_of_total_sum`: arithmetic sums, and the share of the total finite-return sum.

Transitions count only adjacent labels that are both valid. If a row has no transitions, its probabilities are `None`; the function does not invent a distribution.

```python
labels = rflux.regimelab.volatility_regime(market_returns).labels
report = rflux.regime_attribution(market_returns, labels)
print(report["states"]["0"]["annualized_volatility"])
print(report["transitions"]["counts"])
```

### Interpretation

Regime attribution is descriptive, not causal. State sample sizes can differ greatly, and `share_of_total_sum` is not additive across a compounded path; treat it as an arithmetic-period diagnostic. If labels come from an HMM, use the causal `filtered()` or `filtered_states()` output. `smoothed()` and Viterbi use the full sample and are not real-time labels.

## 3. Correlation-network risk diagnostics

`percolab.network_from_correlation(corr, threshold)` turns a correlation matrix into a non-negative adjacency graph and reports local and global structure. The threshold is a modelling choice: an asset pair becomes an edge when the correlation is at least `threshold`, and edge weights are the non-negative correlations themselves.

The `CorrelationNetwork` result contains:

- `components`: labels, sizes, component count and the giant fraction;
- `strength`, `eigenvector`, `clustering`, `average_clustering`;
- `path_stats`: global efficiency, average path length over reachable pairs, diameter, reachable-pair count and a connectivity flag;
- `core_numbers` and `max_core`: k-core shells, which separate the densely connected core from the periphery;
- `transitivity`: global clustering;
- `assortativity`: degree correlation, NaN for regular graphs;
- `mst`, `density` and `n_edges`.

Separately, `percolab.bridges(adjacency)` and `percolab.articulation_points(adjacency)` identify cut edges and cut nodes, and `percolab.truss_numbers(adjacency)` gives triangle-based edge cohesion. They accept any validated adjacency matrix and are not recomputed inside `CorrelationNetwork`.

`mst_clusters(network.distance, cutoff)` cuts the MST at a distance cutoff to produce single-linkage clusters; `modularity(adjacency, labels)` scores an existing sector or cluster partition. Neither function chooses the number of communities: the cutoff or labels carry that decision.

Percolation and removal curves are stress descriptions. `bond_percolation` and `site_percolation` scan occupation probabilities with a fixed seed; the weighted variants and the general `percolation_scan` separate the candidate graph from edge or node occupation weights. `targeted_removal` removes nodes in a supplied or metric-based order and reports the giant fraction, finite-cluster susceptibility and component count. `core_percolation` is the deterministic k-core curve, and `cluster_size_distribution` reports the mean number of clusters of each size at one probability.

```python
import numpy as np
import rflux

corr = rflux.qflux.correlation_matrix(returns)
net = rflux.percolab.network_from_correlation(corr, 0.45)
print(net.components.giant_fraction, net.path_stats.efficiency, net.max_core)

scan = rflux.percolab.bond_percolation(
    net.adjacency, np.linspace(0.0, 1.0, 21), repetitions=200, seed=7
)
print(scan.critical_probability(), scan.max_susceptibility(), scan.giant_area())

curve = rflux.percolab.targeted_removal(
    net.adjacency, strategy="eigenvector", checkpoints=10
)
print(curve.giant_fraction)
print(rflux.percolab.bridges(net.adjacency).shape[0])
print(rflux.percolab.truss_numbers(net.adjacency).max())

sectors = np.array([0, 0, 1, 1, 2, 2])  # one label per asset
print(rflux.percolab.modularity(net.adjacency, sectors))
```

### Interpretation

- Choose the threshold before looking at the result. Edge count and giant fraction are monotone in the threshold; a different threshold is a different graph, not a robust finding.
- Monte Carlo curves carry sampling error and depend on the explicit seed. Increase `repetitions` and check stability across seeds before reading a crossing point.
- `critical_probability()` is the scanned probability with maximum finite-cluster susceptibility. It is a finite-size descriptive estimate, not an extrapolated thermodynamic threshold.
- `targeted_removal` is a static attack curve; it does not adapt the order after each removal. `removal_order` supports strength, degree, eigenvector, betweenness and core-number strategies.
- Correlation is not causation, and a network snapshot does not model liquidity, borrow costs or trading capacity. Treat these diagnostics as conditioning variables and stress scenarios.

## 4. Command line

`rflux risk` converts a price CSV to returns and calls `portfolio_risk`:

```bash
PYTHONPATH=python python -m rflux risk examples/data/sample_prices.csv \
  --weights 0.2,0.2,0.2,0.2,0.2

PYTHONPATH=python python -m rflux risk examples/data/sample_prices.csv \
  --weights 1,-1,0,0,0 --format json
```

Weights follow the CSV data-column order after the date column. Text output lists each asset's weight, annualised volatility, component contribution and share. JSON output also includes the covariance matrix, method metadata and portfolio metrics.

`rflux analyze` remains the command for asset statistics, correlation networks and HMMs; its network report now includes efficiency, path length, diameter, transitivity, assortativity and maximum core number. `risk` only handles the weighted portfolio. Neither command needs pandas.

## 5. Reproducibility

The top-level `method` and `input` fields record the estimator, `ddof`, annualisation, risk-free rate, confidence, complete sample count and discarded rows. Save the full report when reproducing a result; do not keep only rounded returns.

These applications do not model borrow fees, transaction costs, taxes, capacity, limit moves or liquidity. The portfolio return is an arithmetic combination under supplied weights, not a tradable net asset value.


## Research advice

`rflux advice` produces an auditable rebalance proposal, not personalised
investment advice:

```bash
rflux advice prices.csv --weights 0.2,0.2,0.2,0.2,0.2 --max-weight 0.35
```

It starts from HRP weights, adjusts them with 20/60-day momentum, trend and
RSI, then scales gross exposure by realised volatility and an optional HMM
stress state.  The report contains:

- current/target weights, change, signal score and reason per asset;
- cash weight, gross exposure and turnover;
- target historical volatility, max drawdown and VaR/CVaR;
- data/model errors and an explicit non-personalised-advice disclaimer.

It is long-only and unlevered.  Transaction costs, taxes, liquidity and lot
sizes remain the caller's responsibility.
