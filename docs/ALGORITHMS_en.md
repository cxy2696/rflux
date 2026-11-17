# Algorithm matrix and extension contracts

**English** · [中文](ALGORITHMS_cn.md)

rflux 0.7 introduces a C++ algorithm layer on top of the numerical core.  The
goal is not to add another pile of indicators: every algorithm has an explicit
interface, a documented input/output contract, a native fast path and a
research-layer fallback where practical.

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

**Next layer:** analyst-estimate revisions, PE/PB/ROE fundamentals, LLM-generated
factor hypotheses and evolutionary formula search are data/adapter concerns;
the factor interface is ready for them but the data contracts are not bundled.

## Module 2: regime and tail risk

- Existing diagonal Gaussian HMM: filtering, smoothing, Viterbi, forecasts,
  state alignment and walk-forward scoring.
- `rflux.regimelab.sticky_regime`: hysteresis plus a minimum dwell time; a
  challenger needs an entry probability and the incumbent must fall below an
  exit probability before the label changes.  This reduces one-bar turnover.
- `rflux.risk.tail_dependence`: empirical lower/upper pairwise tail dependence,
  capturing extreme co-movement beyond Pearson correlation.
- Existing `percolab` network/percolation tools cover risk-contagion paths.

The C++ `HmmRegimeDetector` adapts the existing HMM to `BaseRegimeDetector`.

## Module 3: portfolio construction

`rflux.portfoliolab` provides:

- `hrp_weights`: correlation distance, single-linkage tree, quasi-diagonal
  order and recursive bisection.  No covariance inversion; long-only and
  unit-gross.
- `nco_weights`: nested clustered risk parity.  Assets are grouped by
  correlation distance, each group is HRP-allocated and groups receive inverse
  cluster-variance budgets.  This is the NCO-style clustering layer; a full
  constrained NCO convex stage is a documented next step.
- `risk_parity_weights`: diagonal inverse-variance baseline.
- `rebalance_targets`: hysteresis on target drift, useful with a sticky regime
  gate to avoid unnecessary turnover.

## Module 4: robust machine learning

`rflux.mllab` provides:

- `huber_regression`: iteratively reweighted least squares with a robust MAD
  scale.  A C++ implementation lives in
  [`include/rflux/mllab/huber.hpp`](../include/rflux/mllab/huber.hpp) and is
  exposed through the native bridge; NumPy is the fallback.
- `cvar_loss`: historical CVaR of losses, useful as a model-selection or
  portfolio objective diagnostic.
- `graph_smooth`: degree-normalised diffusion of a node signal as a
  lightweight graph regulariser.  It is not a trained GNN.

**Next layer:** temporal GNNs, graph attention, message passing over supply
chains and full CVaR-regularised training belong behind the `BasePredictor`
contract.

## Module 5: execution and microstructure

`rflux.execution` provides:

- `almgren_chriss`: optimal deterministic liquidation with linear temporary and
  permanent impact, risk aversion and a requested slice count.  Returns trades,
  expected cost, cost variance and `kappa`.
- `order_book_metrics`: level-1 spread, mid, microprice and size imbalance from
  caller-supplied top-of-book snapshots.

**Next layer:** queue position, tick-size constrained child orders, dark pools,
venue routing and QML/quantum pricing are venue- or hardware-specific and are
not included in this release.

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
