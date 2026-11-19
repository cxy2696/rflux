# C++ core contracts: market data, predictors, execution and percolation

**English** · [中文](CORE_cn.md)

The C++ layer follows three design rules:

1. **No Eigen in the public boundary.** The C ABI and pybind11 take plain
   contiguous buffers. Eigen would add a large header dependency, alignment rules
   and ABI/lifetime questions. The core uses `MarketSnapshotView`, `MatrixView`,
   `OrderBookView` and standard ownership containers.
2. **Predictor and execution contracts carry context.** A predictor has `fit`
   and `fitted`; an execution request carries symbols, target/current weights,
   portfolio value and timestamp. A bare weight vector and one anonymous order
   book are not enough to derive orders.
3. **Factories take explicit specs and are thread-safe.** Built-in factors need
   a window or scale; a zero-argument `unique_ptr<BaseFactor>()` creator cannot
   carry that. `FactorFactory` consumes a `FactorSpec`, rejects duplicate names
   and copies creators under a mutex before invoking them.

The Python package is a thin consumer of this core. Data adapters and CLI
formatting stay in Python, but every algorithm, backtest, graph, report
statistic and HMM computation runs in C++ behind the pybind11 extension or the
C ABI. The extension is required; there is no NumPy fallback path.

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
- `estimate_threshold()` calibrates `p_c` on a configurable probability grid
  and interpolates the first giant-fraction crossing; `percolation_threshold`
  is NaN until calibration.  This avoids pretending that every `analyze(p)`
  call can derive a stable critical threshold in a hot path.
- `rflux::risk::PreTradeRiskControl` can gate an order on position, drawdown,
  turnover and a systemic-percolation flag.

The Monte Carlo kernel uses the internal deterministic fork-join helper in
`src/internal/parallel.hpp`.  Probability points and replicate seeds are split
into static index blocks, and every block keeps its original random stream and
accumulation order, so results are bit-for-bit identical for any worker count.
`RFLUX_THREADS=N` caps the workers and `RFLUX_THREADS=1` forces serial
execution.  Threshold calibration interpolates the first grid crossing; a fully
adaptive bisection search remains a next step.  The union-find complexity per trial is O(n + m α(n)).

## Buffer-first numerical kernels

Every qflux kernel has two overloads in its public header: `foo_into(...,
double* out)` writes exactly `n` values and allocates nothing, while `foo(...)`
allocates a result vector and forwards to the buffer version.  The C ABI and
the Python bridge call the buffer version, so a call never materialises a
temporary result vector: C writes straight into the caller's array and
pybind11 writes straight into the NumPy output (`series_into` is therefore
truly zero-copy).  `out` must not alias an input and must have room for `n`
values; arguments are validated before the first write.

`rolling_quantile` chooses its order-statistic structure from the window
length.  Windows up to 1024 values keep a sorted buffer (binary search plus a
contiguous shift, no global coordinate compression); longer windows use the
Fenwick tree over compressed coordinates.  Both paths return the same Type-7
statistic.

The research kernels follow the same ownership rule.  `factorlab/backtest.cpp`
takes row-major factor and forward-return panels, splits each cross-section with
`np.array_split` sizes and writes result vectors; the C ABI copies them into
caller buffers.  `regimelab/alignment.cpp` returns a state permutation from the
Hungarian assignment and owns no model state, so it works with parameters read
from a live model or a saved snapshot.  `portfoliolab/advice.cpp` and
`regimelab/attribution.cpp` follow the same pattern for the research reports:
they return numeric structs, and the Python layer only formats JSON.

## Deterministic parallelism

`detail::parallel_for(count, fn, max_threads)` is the only threading primitive
in the core.  It partitions `[0, count)` statically, never depends on scheduling
state and re-throws the first worker exception.  It drives percolation scans
(parallel over probability points), cluster-size sampling (private per-worker
counters reduced in worker order), HMM Baum-Welch restarts (parallel over
`n_init`, capped at four concurrent restarts to bound memory) and correlation
matrices (parallel over assets on both the complete-panel and
pairwise-complete paths).  Small inputs skip the fork-join because thread
setup would cost more than the work, and nested calls fall back to serial
execution so a kernel never oversubscribes the machine.

## Build optimization

CMake enables interprocedural optimization (IPO/LTO) for the core, C ABI,
native extension, test and example targets when the compiler supports it.
Release and RelWithDebInfo builds use it; Debug and sanitizer builds skip it.
Pass `-DRFLUX_ENABLE_IPO=OFF` to compare builds or to avoid long link times.

## C ABI coverage

The public kernels are reachable from C: factor panels/IC/turnover, backtests,
HRP/NCO and risk-parity weights, portfolio risk, rebalance gating, performance
summaries, tail dependence, CVaR, graph smoothing, order-book metrics,
Almgren-Chriss plans, Huber regression, site-percolation analysis and
threshold, HMM state matching and the pre-trade gate.

## Shared code

- Percolation reuses the shared `UnionFind` and `mix_seed` helpers.
- HMM k-medoid nearest-center selection is one helper.
- Huber computes each residual pass once and reuses one linear-prediction helper.
- Factor pybind11 bindings share `make_factor_context` / `compute_factor_panel`.
- SQLite cache writes use array zips instead of `iterrows()`.
- `SlidingBlocks<Ops>` is the only two-stack merging queue; `Moments1Ops` and
  `Moments2Ops` supply the per-arity single/merge operations.
- `for_each_complete_window` owns the warm-up, expiry and missing-data rules for
  every rolling kernel, and both `rolling_quantile` backends reuse it.
- The rolling correlation, covariance and beta kernels are one
  `rolling_pair_stat_into` implementation with a statistic selector; Sharpe and
  Sortino share one `performance_ratio` body.
- The C ABI HMM entry points share `guarded_model_vector` / `guarded_model_i32`,
  and the C++ model has one `validate_inference` / `require_fit_diagnostics`
  guard.  The Python bridge has one `run_released`, `filled_array`,
  `filled_triple` and `array_from` path.
- The Python layer chooses the backend through `native_or` / `native_call` and
  `GaussianHMM._native_or_abi`; report metadata, CLI report finishing,
  percolation argument marshalling and probability-matrix reads each have one
  helper.
- jscpd reports zero exact clones for `src/`, `include/`, `python/rflux/` and
  `tests/` at both the default 5-line/50-token and the stricter
  5-line/40-token setting.  The bilingual documentation mirrors are the only
  expected repeated blocks.
