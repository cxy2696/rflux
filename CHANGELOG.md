# Changelog

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
