# C++ core contracts: market data, predictors, execution and percolation

This page describes the 0.7 C++ extension layer.  The design differs from an Eigen-based header in three places:

1. **No Eigen in the public boundary.**  rflux already exposes plain contiguous
   buffers to the C ABI and pybind11.  Eigen would add a heavy header dependency,
   alignment rules and ABI/lifetime questions.  The core uses `MarketSnapshotView`,
   `MatrixView`, `OrderBookView` and standard ownership containers instead.
2. **Predict/execute contracts carry enough context.**  A predictor has `fit`
   and `fitted`; an execution request carries symbols, target/current weights,
   portfolio value and timestamp, because orders cannot be derived from a bare
   weight vector and one anonymous order book.
3. **Factories are parameterized and thread-safe.**  Built-in factors need a
   window/scale, so a zero-argument `unique_ptr<BaseFactor>()` creator is not
   sufficient.  `FactorFactory` consumes a `FactorSpec`, rejects duplicate names
   and copies creators under a mutex before invoking them.

## Cross-module structures

[`include/rflux/core/market.hpp`](../include/rflux/core/market.hpp):

- `MarketData` / `MarketSnapshotView`: owning and zero-copy OHLCV snapshots.
- `OrderBook` / `OrderBookView`: level-2 top-of-book levels.
- `MatrixView`: row-major feature matrix.
- `PredictionResult`: expected returns plus confidence.
- `RiskConstraints`, `Order`, `Side`, `ExecutionRequest`.

All views are caller-owned and must outlive the algorithm call.  Validation
helpers reject null buffers, empty dimensions and malformed books.

## Predictor and execution interfaces

- `mllab::BaseForecastModel`: `fit(MatrixView, target)` plus
  `predict(MatrixView, regime) -> PredictionResult`; `fitted()` is explicit.
- `execution::BaseExecutionAlgo`: `execute(ExecutionRequest, OrderBookView)`
  returns child `Order`s.  The request is single-symbol; a multi-asset parent
  order engine must split first.

The lower-level `core::BasePredictor` remains the narrow feature-to-target
regressor used by `HuberRegressor`, so existing models do not need a fake
regime argument.

## Factor registry

[`include/rflux/factorlab/factory.hpp`](../include/rflux/factorlab/factory.hpp):

```cpp
rflux::factorlab::register_builtin_factors();
rflux::factorlab::FactorSpec spec;
spec.window = 20;
auto factor = rflux::factorlab::FactorFactory::instance()
                  .create("momentum", spec);
```

`AlgorithmRegistry<Interface, Args...>` is reused by every algorithm family:

| Factory | Spec | Built-in |
| --- | --- | --- |
| `FactorFactory` | `FactorSpec` | momentum, volume_reversal, vwap_deviation, amihud |
| `AllocatorFactory` | none | hrp |
| `PredictorFactory` | `PredictorSpec` | huber |
| `ExecutionPlannerFactory` | `ExecutionPlannerSpec` | almgren_chriss |
| `PercolationDetectorFactory` | `PercolationSpec` | site_percolation |
| `RegimeDetectorFactory` | `RegimeSpec` | gaussian_hmm |

Registration is thread-safe, duplicate names fail unless `replace=true`, and
creation does not hold the registry lock while a constructor runs.

## Financial networks and percolation

[`include/rflux/percolab/detector.hpp`](../include/rflux/percolab/detector.hpp):

- `FinancialNetwork` is CSR (`row_offsets`, `column_indices`, `edge_weights`)
  with an optional symbol-to-index map, avoiding a dense 5000 x 5000 matrix.
- `make_network(dense, n)` converts a dense adjacency matrix.
- `SitePercolationDetector` runs random site percolation with a deterministic
  seed and union-find components.  It reports giant-component fraction, standard
  deviation, finite-cluster susceptibility, largest component and repetitions.
- `estimate_threshold()` calibrates `p_c` on a configurable probability grid;
  `percolation_threshold` is NaN until calibration.  This avoids pretending that
  every `analyze(p)` call can derive a stable critical threshold in a hot path.
- `rflux::risk::PreTradeRiskControl` can gate an order on position, drawdown,
  turnover and a systemic-percolation flag.

The Monte Carlo kernel is currently serial and uses a fixed grid for `p_c`.
OpenMP/TBB acceleration and adaptive threshold search are explicit next steps,
The union-find complexity per trial is O(n + m α(n)).

## C ABI coverage

The 0.7 kernels are reachable from C: factors, HRP weights and leaf order,
Almgren-Chriss plans, Huber regression, site-percolation analysis and
threshold, and the pre-trade gate.  Python uses those C functions when the
pybind11 extension is unavailable instead of keeping a second NumPy
implementation.

## Duplication cleanup

- Percolation reuses the shared `UnionFind` and `mix_seed` helpers.
- HMM k-medoid nearest-center selection is one helper.
- Huber computes each residual pass once and reuses one linear-prediction helper.
- Factor pybind11 bindings share `make_factor_context` / `compute_factor_panel`.
- SQLite cache writes use array zips instead of `iterrows()`.
