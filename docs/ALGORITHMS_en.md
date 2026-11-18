# Algorithm matrix and extension contracts

**English** · [中文](ALGORITHMS_cn.md)

rflux 0.7 adds a C++ algorithm layer on top of the numerical core. Every
algorithm has a typed interface, a documented input/output contract, a native
implementation, and a Python fallback where needed.

## Extension contracts

The underlying market-data views, predictor/execution interfaces, registry and
percolation contracts are described in [CORE_en.md](CORE_en.md).

The public contracts live in
[`include/rflux/core/algorithm.hpp`](../include/rflux/core/algorithm.hpp):

| Interface | Responsibility |
| --- | --- |
| `BaseAlgorithm` | stable name/version and virtual destruction |
| `BaseFactor` | turn a `FactorContext` into a row-major factor panel |
| `BaseRegimeDetector` | fit a state model and expose states/probabilities |
| `BaseAllocator` | map a covariance matrix to long-only weights |
| `BasePredictor` | fit/predict a feature matrix |
| `BaseExecutionPlanner` | map an order into a deterministic execution plan |

All inputs and outputs are caller-owned contiguous buffers.  The C++ core does
not parse files, load market data or depend on Python.  TOML/JSON/YAML parsing,
provider adapters and language bindings stay at the composition edge.

## Module 1: factor engineering

All four factors derive from `PanelFactor`, which validates the panel and
centralizes the asset/time traversal. Each factor only supplies a cell function
and a trailing-return helper, so adding a factor no longer duplicates output
allocation or warm-up handling.

C++ classes in [`include/rflux/factorlab/factors.hpp`](../include/rflux/factorlab/factors.hpp):

| Factor | Definition | Warm-up |
| --- | --- | --- |
| `MomentumFactor(window)` | `close[t] / close[t-window] - 1` | `t >= window` |
| `VolumeReversalFactor(window)` | `-ret(window) * volume[t] / mean(volume)` | `t >= window` |
| `VwapDeviationFactor(window)` | `(close[t] - VWAP_window) / VWAP_window` | `t >= window-1` |
| `AmihudFactor(window, scale)` | `scale * mean(abs(ret) / dollar_volume)` | `t >= window` |

All panels are `(n_obs, n_assets)`.  A cell is NaN when a required input is
missing or during warm-up.  The volume factors require non-negative volume.

Python: `rflux.factorlab` exposes `momentum`, `volume_reversal`,
`vwap_deviation`, `amihud`, `information_coefficient` (Pearson or rank IC per
date) and `factor_turnover`.

**Future work:** analyst-estimate revisions, PE/PB/ROE fundamentals, LLM-generated
factor hypotheses and evolutionary formula search need data adapters. The factor
interface is ready; the data contracts are not bundled.

## Module 2: regime and tail risk

- Existing diagonal Gaussian HMM: filtering, smoothing, Viterbi, forecasts,
  state alignment and walk-forward scoring.  `fit_hmm_snapshot` centralises
  one-shot fitting, closes the stateful model, and returns detached latest
  filtered/forecast arrays for reports.
- `rflux.regimelab.sticky_regime`: hysteresis plus a minimum dwell time; a
  challenger needs an entry probability and the incumbent must fall below an
  exit probability before the label changes.  This reduces one-bar turnover.
- `rflux.risk.tail_dependence`: empirical lower/upper pairwise tail dependence,
  capturing extreme co-movement beyond Pearson correlation.  Each pair averages
  the two finite-sample conditionals `P(i | j)` and `P(j | i)` so the reported
  matrix is symmetric.
- Existing `percolab` network/percolation tools cover risk-contagion paths.

The C++ `HmmRegimeDetector` adapts the existing HMM to `BaseRegimeDetector`.

## Module 3: portfolio construction

`rflux.portfoliolab` provides:

- `hrp_weights`: correlation distance, single-linkage tree, quasi-diagonal
  order and recursive bisection.  No covariance inversion; long-only and
  unit-gross.
- `nco_weights`: nested clustered risk parity.  Assets are grouped by
  correlation distance, each group is HRP-allocated and groups receive inverse
  cluster-variance budgets. A full constrained NCO convex stage is future work.
- `risk_parity_weights`: diagonal inverse-variance baseline.
- `rebalance_targets`: drift-threshold rebalancing gated independently per
  portfolio row, useful with a sticky regime gate to avoid unnecessary
  turnover.

## Module 4: robust machine learning

`rflux.mllab` provides:

- `huber_regression`: iteratively reweighted least squares with a robust MAD
  scale.  A C++ implementation lives in
  [`include/rflux/mllab/huber.hpp`](../include/rflux/mllab/huber.hpp) and is
  exposed through the native bridge; NumPy is the fallback.
- `cvar_loss`: historical CVaR of losses, useful as a model-selection or
  portfolio objective diagnostic.
- `graph_smooth`: degree-normalised diffusion of a node signal as a
  graph smoothing primitive. Model training is outside this module.

**Future work:** temporal GNNs, graph attention, supply-chain message passing and
CVaR-regularised training belong behind the `BasePredictor` contract.

## Module 5: execution and microstructure

`rflux.execution` provides:

- `almgren_chriss`: optimal deterministic liquidation with linear temporary and
  permanent impact, risk aversion and a requested slice count.  Returns trades,
  expected cost, cost variance and `kappa`.  Temporary cost uses the
  continuous-time convention $\eta\sum_j(n_j/\tau)^2\tau$; with a fixed
  coefficient the risk-neutral cost is $\eta Q^2/T$, independent of the slice
  count.
- `order_book_metrics`: level-1 spread, mid, microprice and size imbalance from
  caller-supplied top-of-book snapshots.

**Future work:** queue position, tick-size constraints, dark pools, venue routing
and quantum pricing depend on the venue or hardware and are not in this release.

## Traceability and verification

Each public family has one implementation entry point, an optional stable C ABI
contract, Python wrappers, and named regression tests.  The mapping below is
the audit trail from formula to code:

| Family | C++ implementation | C ABI | Python API | Primary tests |
| --- | --- | --- | --- | --- |
| Series, rolling windows | [`src/qflux/window.cpp`](../src/qflux/window.cpp) | `rflux_rolling_*`, `rflux_sma`, `rflux_ema` | `qflux.rolling_*`, `qflux.sma`, `qflux.ema` | [`test_qflux.py`](../tests/python/test_qflux.py), [`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| Indicators | [`src/qflux/indicators.cpp`](../src/qflux/indicators.cpp) | `rflux_returns`, `rflux_rsi`, `rflux_atr`, `rflux_macd`, `rflux_bollinger`, `rflux_drawdown` | `qflux.returns`, `qflux.rsi`, `qflux.atr`, `qflux.macd`, `qflux.bollinger`, `qflux.drawdown` | [`test_qflux.py`](../tests/python/test_qflux.py), [`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| Statistics and ranking | [`src/qflux/stats.cpp`](../src/qflux/stats.cpp), [`ranking.cpp`](../src/qflux/ranking.cpp), [`performance.cpp`](../src/qflux/performance.cpp) | `rflux_describe`, `rflux_quantile`, `rflux_pearson`, `rflux_sharpe`, `rflux_sortino`, `rflux_var_cvar`, `rflux_rank`, `rflux_zscore`, `rflux_winsorize`, `rflux_neutralize`, `rflux_equity_curve`, `rflux_performance_summary` | `qflux.describe`, `qflux.quantile`, `qflux.pearson`, `qflux.sharpe`, `qflux.sortino`, `qflux.var_cvar`, `qflux.rank`, `qflux.zscore`, `qflux.winsorize`, `qflux.neutralize` | [`test_qflux.py`](../tests/python/test_qflux.py), [`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| Gaussian HMM | [`src/regimelab/hmm.cpp`](../src/regimelab/hmm.cpp) | `rflux_hmm_*` | `regimelab.GaussianHMM`, `fit_hmm`, `fit_hmm_snapshot` | [`test_hmm_regime.py`](../tests/python/test_hmm_regime.py), [`test_native.py`](../tests/python/test_native.py) |
| Rule regimes | [`src/regimelab/regime.cpp`](../src/regimelab/regime.cpp) | `rflux_volatility_regime`, `rflux_trend_regime`, `rflux_composite_regime` | `regimelab.volatility_regime`, `trend_regime`, `composite_regime`, `sticky_regime` | [`test_hmm_regime.py`](../tests/python/test_hmm_regime.py) |
| Network metrics | [`src/percolab/network.cpp`](../src/percolab/network.cpp) | `rflux_network_*`, `rflux_pagerank`, `rflux_corr_to_distance`, `rflux_mst_clusters` | `percolab.centrality`, `pagerank`, `betweenness`, `clustering`, `path_statistics`, `core_numbers`, `truss_numbers`, `mst_*` | [`test_percolab.py`](../tests/python/test_percolab.py), [`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| Percolation and contagion | [`src/percolab/percolation.cpp`](../src/percolab/percolation.cpp), [`detector.cpp`](../src/percolab/detector.cpp) | `rflux_*percolation*`, `rflux_targeted_removal`, `rflux_risk_diffusion` | `percolab.site_percolation`, `bond_percolation`, `percolation_scan`, `cluster_size_distribution`, `targeted_removal`, `core_percolation`, `risk_diffusion` | [`test_percolab.py`](../tests/python/test_percolab.py), [`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| Factors | [`src/factorlab/factors.cpp`](../src/factorlab/factors.cpp), [`src/factorlab/evaluation.cpp`](../src/factorlab/evaluation.cpp) | `rflux_factor_*`, `rflux_factor_ic`, `rflux_factor_turnover` | `factorlab.momentum`, `volume_reversal`, `vwap_deviation`, `amihud`, `information_coefficient`, `factor_turnover` | [`test_algorithms.py`](../tests/python/test_algorithms.py), [`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| Portfolio construction | [`src/portfoliolab/hrp.cpp`](../src/portfoliolab/hrp.cpp), [`src/portfoliolab/risk.cpp`](../src/portfoliolab/risk.cpp) | `rflux_hrp_weights`, `rflux_hrp_order`, `rflux_risk_parity_weights`, `rflux_nco_weights`, `rflux_rebalance_targets`, `rflux_portfolio_risk` | `portfoliolab.hrp_weights`, `nco_weights`, `risk_parity_weights`, `rebalance_targets` | [`test_algorithms.py`](../tests/python/test_algorithms.py), [`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| Robust ML | [`src/mllab/huber.cpp`](../src/mllab/huber.cpp), [`src/mllab/robust.cpp`](../src/mllab/robust.cpp) | `rflux_huber_regression`, `rflux_cvar_loss`, `rflux_graph_smooth` | `mllab.huber_regression`, `cvar_loss`, `graph_smooth` | [`test_algorithms.py`](../tests/python/test_algorithms.py), [`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| Execution and risk | [`src/execution/almgren_chriss.cpp`](../src/execution/almgren_chriss.cpp), [`src/execution/microstructure.cpp`](../src/execution/microstructure.cpp), [`src/risk/pretrade.cpp`](../src/risk/pretrade.cpp), [`src/risk/tail_dependence.cpp`](../src/risk/tail_dependence.cpp) | `rflux_almgren_chriss_plan`, `rflux_order_book_metrics`, `rflux_pretrade_check`, `rflux_tail_dependence` | `execution.almgren_chriss`, `execution.order_book_metrics`, `risk.tail_dependence` | [`test_algorithms.py`](../tests/python/test_algorithms.py), [`test_core_gtest.cpp`](../tests/cpp/test_core_gtest.cpp) |
| Research reports | Python composition (`analysis.py`, `advice.py`, `apps.py`, `backtest.py`); `portfolio_risk` and metrics call C++ | — | `analyze_prices`, `trade_advice`, `portfolio_risk`, `regime_attribution`, `performance_summary`, `backtest.*` | [`test_advice.py`](../tests/python/test_advice.py), [`test_tools.py`](../tests/python/test_tools.py), [`test_apps.py`](../tests/python/test_apps.py) |

`scripts/verify_algorithms.py` is the executable audit: it exercises all public
families, checks names/shapes/warm-ups/ranges and independent references where
practical, records hot-kernel timings, and writes a timestamped JSON report to
`artifacts/validation/`.  `make algo-check` runs it twice, once through the
native extension (when installed) and once through the ctypes C ABI fallback.
The matrix lives next to the algorithms. A new public family needs a row and a
test before review.

## Engineering rules for new algorithms

1. Implement the math in C++ under `src/` and expose it through a contract in
   `include/rflux/`.
2. Add a GoogleTest case with analytic or independent brute-force reference
   values; test edge cases (NaN, zero volume, zero variance, empty panels).
3. Add a native pybind11 binding for the Python fast path, release the GIL for
   nontrivial work, and accept/return contiguous float64 arrays.
4. Provide a NumPy fallback when the algorithm is part of the public Python API,
   and test native/fallback parity.
5. Keep the C ABI additive if a kernel becomes part of the cross-language
   contract.  Do not add file I/O or Python types to the C++ core.
6. Document warm-up, causality, scaling and the exact estimator definition.
