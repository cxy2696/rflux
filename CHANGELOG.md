# Changelog

## Unreleased

Performance and architecture work on the native core; the C ABI stays `1` and
existing signatures, layouts and result semantics are unchanged.

- qflux kernels now have buffer-first `*_into` overloads.  The C ABI writes
  directly into caller arrays and the pybind11 bridge writes directly into the
  NumPy result, so a call no longer allocates a temporary `std::vector` and
  copies it (this makes `series_into` zero-copy).
- `rolling_quantile` switches between a sorted window (up to 1024 values) and
  the Fenwick tree (longer windows).  Short windows no longer sort all
  coordinates, which is several times faster in the common factor case.
- Added `internal/parallel.hpp`, a deterministic static-partition fork-join
  helper with exception propagation, nested-call serialisation and an
  `RFLUX_THREADS` cap.  It parallelises percolation scans, cluster-size
  sampling, HMM restarts and complete-panel correlation matrices while keeping
  results identical for any worker count.
- HMM emission log-normalisers and inverse variances are computed once per
  M-step instead of once per observation, and Baum-Welch restarts run in
  parallel (capped at four to bound memory).
- Sliding moments append singletons through a specialised merge (one reciprocal
  instead of two divisions per observation), and the general block merge also
  reuses one reciprocal.
- Scalar quantiles, historical VaR/CVaR and winsorisation select order
  statistics with `std::nth_element` instead of sorting the sample (about
  2.6-2.7x faster for 1M values, same Type-7 results).
- The minimum spanning tree switched from Kruskal to dense Prim.  It no longer
  materialises or sorts all `n(n-1)/2` candidate edges: time drops from
  `O(n^2 log n)` to `O(n^2)` and auxiliary memory from `O(n^2)` to `O(n)`.
  Local timings for random complete graphs were 1.23 ms -> 0.27 ms at n=200,
  6.40 ms -> 0.42 ms at n=400 and 20.83 ms -> 1.90 ms at n=800.  Output edges
  remain sorted by `(weight, u, v)`.
- CMake now enables interprocedural optimization (IPO/LTO) for core, C ABI,
  native-extension, test and example targets when the toolchain supports it.
  `-DRFLUX_ENABLE_IPO=OFF` restores the old build; Debug builds skip IPO. The
  internal exception classes use default symbol visibility so pybind11's error
  translator still sees one RTTI type when the core is linked with LTO.
- Complete-panel correlation matrices precompute each asset's mean and variance
  and use one covariance pass per pair, in parallel across assets; a
  100-asset x 2000-row panel drops from about 30 ms to 2.6 ms in the local
  benchmark, while the pairwise-complete path stays unchanged.
- Removed the remaining duplicated native blocks: one generic two-stack
  `SlidingBlocks` queue for both moment arities, one complete-window traversal
  for every rolling kernel and quantile backend, one pairwise-statistic kernel
  for correlation/covariance/beta, one Sharpe/Sortino body, shared
  correlation-cell iteration and shared HMM guards.
- The Python layer now routes every native-vs-C-ABI choice through
  `native_or`/`native_call` and one `GaussianHMM._native_or_abi`, replacing the
  per-function `try/except` blocks; report metadata, CLI report finishing,
  percolation argument marshalling and probability matrices each have one
  helper.
- Tests share graph fixtures, HMM fit helpers and synthetic panels. `jscpd`
  reports zero exact clones for `src/`, `include/`, `python/rflux/` and
  `tests/` at 5 lines/40 tokens; only the bilingual documentation mirrors
  repeat. Added an optional `make dup-check` target.
- Python layer: factor IC/turnover, NCO weights, risk parity, rebalance gating,
  tail dependence, CVaR, graph smoothing, order-book metrics, performance
  summaries, equity curves and portfolio risk now run in C++
  (`portfoliolab/hrp.cpp`, `risk/tail_dependence.cpp`) with additive C ABI
  functions and pybind11 entry points. The Python wrappers keep the ctypes
  fallback and dispatch through `native_or`.
- File cleanup: removed the standalone `tests/cpp/test_cpp.cpp` smoke suite;
  its unique describe/ranking/tail-risk assertions moved into the GoogleTest
  suite. CTest now has two targets, and the GoogleTest suite has 30 cases.
- Documentation: replaced generated-sounding prose with direct engineering notes
  in the README, algorithm, core, methods, application, release and example
  pages. `scripts/check_docs.py` now rejects a short list of template markers
  such as `not just`, `deliberately` and `Next layer`.
- Algorithm logic: `trade_advice` now scales exposure by the proposed
  portfolio's ex-ante volatility from its covariance matrix instead of the
  equal-weight market volatility, and its turnover includes the cash leg.
  Momentum and daily-sigma windows use calendar positions rather than skipping
  missing observations.
- Algorithm logic: `risk.tail_dependence` averages the two finite-sample
  conditional probabilities for each pair, so the reported matrix is symmetric. `rebalance_targets` gates each portfolio row independently for 2D
  inputs.
- Algorithm logic: percolation threshold calibration interpolates the first
  giant-fraction crossing instead of snapping to the grid, and the detector
  uses the shared Welford running variance. Volatility-regime classification
  maintains a sorted reference window, yielding identical labels without a
  per-observation sort.
- Python common logic: equal-weight market returns and complete-case
  covariance live in `_common` and are shared by `analysis`, `advice` and
  `apps`; performance-option validation lives in
  `metrics._validated_options`; `fit_hmm_snapshot` centralises one-shot HMM
  fitting and closes the model before returning detached state; `native.py`
  exports from a single list; shared-memory layout validation has one helper;
  `execution` reuses the shared positive-number validator.
- Added an algorithm traceability matrix in `docs/ALGORITHMS_en.md` /
  `_cn.md` and `scripts/verify_algorithms.py`; `make algo-check` runs every
  public family through the native extension and the ctypes C ABI, checks
  shapes/warm-ups/ranges/references, records hot-kernel timings and writes
  timestamped reports under `artifacts/validation/`.
- Packaging hygiene: root-level sdist includes are anchored, so
  `.pytest_cache/README.md` can no longer match the `README.md` pattern;
  caches, logs, `.DS_Store`, artifacts and build directories are excluded.
  Refreshed the wheel/sdist from the final tree.
- Build: CMake links `Threads::Threads`; the developer Makefile passes
  `-pthread` on non-macOS platforms.
- Tests: buffer/vector equivalence for every qflux kernel, brute-force Type-7
  quantile checks across both algorithm paths and windows, buffer overrun
  guards, correlation-matrix equivalence, HMM restart determinism and Monte
  Carlo repeatability; Python tests cover the native `series_into` path.

**English** · [中文](CHANGELOG_cn.md)

## 0.7.1 - 2026-11-18

C ABI remains `1`. This release adds the C++ algorithm contracts, the unified
data layer and the native Python bridge without changing existing C signatures
or struct layouts.

- Added `core/algorithm.hpp`, `core/market.hpp` and `core/registry.hpp`:
  contracts for factors, regimes, allocation, prediction, execution and
  percolation.
- Added `factorlab`, `portfoliolab`, `mllab`, `execution` and `risk`: momentum,
  volume reversal, VWAP, Amihud, HRP/NCO-style allocation, Huber regression,
  Almgren-Chriss execution and pre-trade risk.
- Added the CSR network and union-find site percolation in `percolab`, with
  `p_c` calibration.
- Added `rflux.data`: canonical long OHLCV, CSV/AKShare/yfinance adapters,
  symbol normalisation, SQLite cache and retry wrapper.
- Added the pybind11 bridge, zero-copy arrays, GIL release, CPython 3.12+ abi3
  wheels, configuration-driven HMMs and shared-memory arrays.
- Moved the build to scikit-build-core + CMake/Ninja and added GoogleTest,
  pytest, sanitizers and cibuildwheel.
- Removed duplicated union-find, HMM nearest-centre, Huber residual, factor
  binding and SQLite row-writing code.
- Added the `PanelFactor` base: shared panel validation, output allocation and
  asset/time traversal; the four built-in factors keep only cell logic and a
  common trailing-return helper.
- C++ duplicate blocks reported by `jscpd`: 0.
- Adapted to the AKShare 1.20.46+ `stock_zh_a_hist_em` rename, keeping the old
  SDK fallback.
- Added yfinance `download()` fallback and MultiIndex column handling, plus
  opt-in live provider tests.
- Added C ABI entry points for factors, HRP, Almgren-Chriss, Huber,
  site-percolation detection and pre-trade risk. Python uses the C ABI when the
  native extension is unavailable.
- Removed the Tushare adapter and dependency. The project does not publish to
  PyPI.
- Fixed `largest_component` truncating the mean across Monte Carlo trials; it
  now rounds to the nearest integer.
- Local validation pins CMake's Python to the active virtualenv; `make check`
  prefers `.venv/bin/python` when present.
- Deduplicated the CLI argument definitions and the AKShare endpoint call
  path; removed stale `python/build` output from the working tree.
- Added AKShare ETF routing (`fund_etf_hist_em`) with stock/ETF auto-detection.
- Added `PercolationScan.p_half_giant()` for sparse correlation networks where
  the susceptibility peak degenerates at `p = 0`.
- Fixed `trade_advice` covariance estimation when the first returns row is NaN.
- Corrected the Almgren-Chriss temporary cost to the continuous-time
  convention $\eta\sum(n_j/\tau)^2\tau$; with a fixed coefficient the
  risk-neutral cost no longer grows with the slice count.
- Added `rflux advice`: a research rebalance proposal using HRP base weights,
  momentum/trend/RSI adjustments, volatility targeting, an optional HMM risk
  state, position actions and risk estimates.
- Completed the algorithm-family factories: `AllocatorFactory`,
  `PredictorFactory`, `ExecutionPlannerFactory`,
  `PercolationDetectorFactory` and `RegimeDetectorFactory`, all using the
  parameterised thread-safe registry.

- Made English the default README language, added Chinese mirrors for every Markdown document and added explicit notation/derivation sections for the HMM, HRP, Almgren-Chriss, Huber, percolation and advice formulas.

Earlier versions are available from git history before the 0.7.1 tag.
