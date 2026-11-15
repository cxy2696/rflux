# Changelog

Notable changes are listed here. Numerical definitions, missing-value rules and
causal semantics are called out explicitly when they change.

## 0.4.2 — 2026-11-15

- Added a direct command-line interface: `rflux version`, `rflux doctor` and
  `rflux analyze FILE`. The `analyze` command reads a wide price CSV, computes
  per-asset and equal-weight statistics, a correlation network and an optional
  HMM, and can emit JSON.
- Added the high-level `rflux.analyze_prices()` API and top-level shortcuts such
  as `rflux.sma`, `rflux.returns`, `rflux.fit_hmm`,
  `rflux.correlation_network` and `rflux.performance_summary`.
- Added `examples/data/sample_prices.csv` so the CLI can be tried immediately.
- Split the methods and extension guides into Chinese and English files
  (`METHODS_cn.md`, `METHODS_en.md`, `EXTENDING_cn.md`, `EXTENDING_en.md`) and
  rewrote them for readability.
- Added a cross-sectional factor-screening example and an examples index covering
  factor research, walk-forward HMM, network risk and real data.
- Added visible use-case and direct-use sections to the Chinese and English
  guides.
- Fixed GitHub Markdown math rendering by using `$...$` / `$$...$$` delimiters
  with whitespace boundaries.
- Added `make docs-check` and CI checks for the CLI, examples and Markdown.
- No C ABI break: version 0.4.2 still uses ABI version 1.

## 0.2.0 — 2026-11-15

- Added rolling OLS beta through C++ (`qflux::rolling_beta`), the C ABI
  (`rflux_rolling_beta`) and Python (`qflux.rolling_beta`).
- Added HMM out-of-sample scoring (`GaussianHMM::score`, `rflux_hmm_score`,
  `GaussianHMM.score`) for holdout and walk-forward model selection. This also
  refactored the duplicated filter/EM forward pass into one log-domain routine.
- Added HMM state alignment utilities in Python: `match_states`,
  `align_parameters`, `align_probabilities` and `align_labels`, using a
  Hungarian assignment on standardized mean and log-variance differences.
- Added `filtered_states()` and `conditional_mean()` HMM helpers.
- Added a walk-forward HMM example with out-of-sample scores and aligned labels.
- Added `rflux/rflux.hpp` as a C++ convenience umbrella header.
- Replaced raw sliding sums of squares with stable two-stack mergeable moments for
  rolling std/z-score/correlation/beta/slope/R². Large-offset level series no
  longer lose the small variance to cancellation.
- Fixed eigenvector centrality on bipartite graphs by iterating on `A + I` and
  reporting non-convergence instead of returning a non-eigenvector.
- Hardened Python handles and data paths: HMM copies are rejected in favour of
  `clone()`, CSV dates are deduplicated, an explicitly requested akshare price
  column must exist, empty seed lists are accepted, integer ranges are checked,
  and non-finite forward prices no longer create spurious returns.
- `performance_summary()` now separates observed and elapsed periods, annualises
  over elapsed rows by default, and reports leveraged long-short losses beyond
  -100% with NaN geometric return instead of raising.
- Split documentation into Chinese and English guides; added CI, contributing
  notes and project metadata.
- No C ABI break: existing v1 functions, structures and semantics are unchanged.

## 0.1.0

Initial release.

- Stable C ABI (`RFLUX_ABI_VERSION=1`) with opaque HMM handles, thread-local
  errors and caller-owned output buffers.
- C++17 `qflux`: indicators, rolling statistics, cross-sectional ranking,
  descriptive statistics and tail-risk measures.
- C++17 `regimelab`: diagonal Gaussian HMM with multiple restarts, log-domain
  Baum-Welch, Viterbi, causal filtering, smoothing and state forecasting;
  causal volatility/trend/composite regime rules.
- C++17 `percolab`: correlation distance, threshold components, centrality,
  clustering, MST, bond/site percolation and SIR risk diffusion.
- Python ctypes package with NumPy wrappers, akshare/yfinance/CSV adapters,
  performance metrics, factor IC/quantile portfolios and a transparent
  long-short research backtest.
- C ABI, C++ core and Python unittest suites; C and C++ examples.
