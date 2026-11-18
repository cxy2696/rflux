# rflux C ABI contract

**English** · [中文](ABI_cn.md)

This document only covers `include/rflux/rflux.h`. The C++ headers under
`include/rflux/**/*.hpp` are source interfaces, not a cross-compiler or
cross-version C++ ABI promise.

## 1. Version

```c
#define RFLUX_ABI_VERSION 1u
#define RFLUX_VERSION_MAJOR 0
#define RFLUX_VERSION_MINOR 7
#define RFLUX_VERSION_PATCH 1
```

Runtime checks:

```c
uint32_t rflux_abi_version(void);
const char* rflux_version_string(void);
```

A dynamic binding must check both the ABI number and the version string. The
Python loader rejects anything other than `rflux_abi_version() == 1`.

Version rules:

- Within one `RFLUX_ABI_VERSION`, existing signatures, layouts and semantics
  stay compatible.
- Adding functions or enum values is backward compatible and should raise the
  minor version.
- Removing a function, changing argument/return types or error semantics, or
  extending an ABI-v1 options struct requires a new ABI version.
- A release may add optional functions without forcing old clients to
  recompile.
- `0.7.1` uses ABI `1`. The new C++ algorithm interfaces, Python data layer and
  native bridge are additive.
- Factors, HRP/NCO, tail dependence, Almgren-Chriss, Huber, site-percolation
  detection and the pre-trade gate have C entry points; all new symbols are
  additive.

## 2. Status codes and exceptions

```c
typedef enum rflux_status {
    RFLUX_OK = 0,
    RFLUX_ERR_INVALID_ARGUMENT = 1,
    RFLUX_ERR_OUT_OF_MEMORY = 2,
    RFLUX_ERR_NONFINITE = 3,
    RFLUX_ERR_MODEL_NOT_READY = 4,
    RFLUX_ERR_NUMERICAL = 5,
    RFLUX_ERR_INTERNAL = 6
} rflux_status;
```

C++ exception mapping:

| C++ | C |
| --- | --- |
| normal return | `RFLUX_OK` |
| `rflux::InvalidArgument` / `std::invalid_argument` | `RFLUX_ERR_INVALID_ARGUMENT` |
| `rflux::NonFinite` | `RFLUX_ERR_NONFINITE` |
| `rflux::NotReady` | `RFLUX_ERR_MODEL_NOT_READY` |
| `rflux::NumericalError` | `RFLUX_ERR_NUMERICAL` |
| `std::bad_alloc` | `RFLUX_ERR_OUT_OF_MEMORY` |
| another `std::exception` | `RFLUX_ERR_INTERNAL` |
| unknown exception | `RFLUX_ERR_INTERNAL` |

No exception crosses the C boundary.

`rflux_status_name(int)` accepts any integer; unknown values return
`"RFLUX_ERR_UNKNOWN"`. It does not change the thread error string.

Each thread has its own error buffer:

```c
const char* rflux_last_error(void);
void rflux_clear_error(void);
```

- A successful call clears the calling thread's error string.
- `rflux_last_error()` points to thread-local storage and is valid only until
  the next rflux call on that thread. Do not free or retain it.
- Output buffers are unspecified after a failure and must not be consumed.

## 3. Memory and ownership

- The caller owns every input array; the library only reads it during the call.
- The caller preallocates every output array. The library never allocates,
  adopts or frees caller output buffers.
- `rflux_hmm*` is created by `rflux_hmm_create` and freed by
  `rflux_hmm_destroy`; `destroy(NULL)` is safe.
- Returned `const char*` values are static strings (version, status name) or a
  thread-local error string. The caller must not `free` them.
- Distinct output buffers must not alias. Input/output aliasing happens to work
  for some kernels because results are computed before copying, but this is not
  a contract; keep them separate.
- A failed call may have written part of the output. Treat the whole call as
  failed.

## 4. Output lengths

`K = n_states`, `D = n_features`, `T = n_obs`.

### qflux indicators

| Function | Output length |
| --- | --- |
| `rflux_returns` | n |
| `rflux_sma` / `rflux_rolling_sum` | n |
| `rflux_rolling_std` / `rflux_rolling_min` / `rflux_rolling_max` / `rflux_rolling_zscore` / `rflux_rolling_quantile` | n |
| `rflux_ema` / `rflux_rsi` / `rflux_atr` / `rflux_drawdown` | n |
| `rflux_rolling_corr` / `rflux_rolling_cov` / `rflux_rolling_beta` / `rflux_rolling_slope` / `rflux_rolling_r2` | n |
| `rflux_macd` | three arrays of length n: macd, signal, histogram |
| `rflux_bollinger` | three arrays of length n: middle, upper, lower |

### qflux statistics and cross-section

| Function | Output |
| --- | --- |
| `rflux_describe` | five scalars + one `size_t` |
| `rflux_quantile` / `rflux_pearson` / `rflux_sharpe` / `rflux_sortino` | one scalar |
| `rflux_var_cvar` | var and cvar |
| `rflux_corr_matrix` | `n_assets * n_assets`, row-major |
| `rflux_rank` / `rflux_percentile_rank` / `rflux_zscore` / `rflux_winsorize` / `rflux_neutralize` | n |

### regimelab

| Function | Output |
| --- | --- |
| `rflux_hmm_get_parameters` | start K, transition K², means K·D, variances K·D |
| `rflux_hmm_set_parameters` | same as input |
| `rflux_hmm_viterbi` | T `int32_t` values |
| `rflux_hmm_filtered` / `rflux_hmm_smoothed` | T·K, row-major |
| `rflux_hmm_score` | one scalar: log-likelihood under current parameters |
| `rflux_hmm_forecast` / `rflux_hmm_filter_step` / `rflux_hmm_stationary` / `rflux_hmm_expected_durations` | K |
| `rflux_volatility_regime` | labels n, volatility n |
| `rflux_trend_regime` | labels n, fast_ma n, slow_ma n |
| `rflux_composite_regime` | n |

### 0.7 algorithm layer

| Function | Output |
| --- | --- |
| `rflux_factor_momentum` / `rflux_factor_volume_reversal` / `rflux_factor_vwap_deviation` / `rflux_factor_amihud` | `n_obs * n_assets`, row-major |
| `rflux_factor_ic` | `n_obs` Pearson or Spearman ICs |
| `rflux_factor_turnover` | `n_obs` one-way turnover values |
| `rflux_hrp_weights` | n_assets, long-only and summing to one |
| `rflux_hrp_order` | n_assets `size_t` leaf order |
| `rflux_risk_parity_weights` | n_assets inverse-variance weights |
| `rflux_nco_weights` | n_assets clustered risk-parity weights |
| `rflux_rebalance_targets` | n_rows * n_assets gated weights |
| `rflux_tail_dependence` | n_assets x n_assets symmetric matrix |
| `rflux_almgren_chriss_plan` | trades of length `slices`; cost/variance/kappa are either all provided or all omitted |
| `rflux_huber_regression` | coefficients length `n_features`; intercept, scale, iterations, converged |
| `rflux_cvar_loss` | one scalar |
| `rflux_graph_smooth` | signal length `n` |
| `rflux_order_book_metrics` | four arrays of length `n` |
| `rflux_site_percolation_analyze` | giant, std, susceptibility, largest_component, systemic |
| `rflux_site_percolation_threshold` | one scalar |
| `rflux_pretrade_check` | allowed 0/1 and a NUL-terminated reason |

`largest_component` is the mean largest component size across trials, rounded
to the nearest integer.

### percolab

| Function | Output |
| --- | --- |
| `rflux_corr_to_distance` | n² |
| `rflux_graph_components` | labels n + components + largest |
| `rflux_network_strength` / `rflux_network_centrality` / `rflux_network_clustering` / `rflux_network_betweenness` / `rflux_pagerank` | n |
| `rflux_network_mst` | three edge arrays of at least n-1 entries + n_edges |
| `rflux_bond_percolation` / `rflux_site_percolation` / `rflux_bond_percolation_weighted` / `rflux_site_percolation_weighted` | giant_mean, giant_std and susceptibility per probability |
| `rflux_risk_diffusion` | mean_affected length steps+1, infection_frequency length n |

## 5. HMM options layout

```c
typedef struct rflux_hmm_options {
    size_t struct_size;
    uint32_t max_iter;
    uint32_t n_init;
    double tol;
    double variance_floor;
    uint64_t seed;
    int32_t sort_states;
    int32_t reserved0;
} rflux_hmm_options;
```

- Initialise it with `rflux_hmm_options_init(&options, sizeof(options))`.
- The current implementation requires `struct_size >= sizeof(rflux_hmm_options)`.
- ABI-v1 field order and size are frozen; do not add fields in place.
- Passing `NULL` to `rflux_hmm_fit` uses `max_iter=200`, `n_init=3`,
  `tol=1e-6`, `variance_floor=1e-6`, `seed=42`, `sort_states=1`.

## 6. Thread safety

- Separate `rflux_hmm` handles may be used concurrently.
- One `rflux_hmm` handle is not thread-safe: do not fit, set parameters and run
  inference on the same handle concurrently.
- Pure functions (indicators, statistics, graphs, percolation scans, diffusion)
  hold no shared mutable state and may be called concurrently.
- Error strings are thread-local.
- Randomness appears only in functions that take an explicit seed. Each call
  owns its RNG; there is no global generator.

## 7. Platform exports

- Linux/macOS: `RFLUX_API` expands to `visibility("default")`; the library is
  built with `-fvisibility=hidden`, so only annotated functions are exported.
- Windows: defining `RFLUX_BUILDING_DLL` while building the DLL produces
  `__declspec(dllexport)`; consumers get `__declspec(dllimport)`.
- Static consumers may define `RFLUX_STATIC` to remove the dllimport marker.

## 8. Compatibility checklist

Before a release, check:

1. Exported symbols from `nm -D` / `dumpbin` only grow, never shrink.
2. An old C test program links and runs against the new library.
3. `RFLUX_ABI_VERSION` is raised for every breaking change.
4. `rflux_hmm_options` size and field offsets are unchanged.
5. Error-code meanings are stable, especially `MODEL_NOT_READY` and
   `NONFINITE`.
6. Every output length and NaN convention has a test and a doc entry.

## 9. Python bridge and ABI

The Python package prefers the pybind11 extension `rflux._native`; ctypes uses
the C ABI described here. The extension links `rflux_core` directly, exports no
new C symbols and does not change `rflux.h`, status codes or the options layout.

The extension maps `rflux::InvalidArgument`, `NonFinite`, `NotReady` and
`NumericalError` to `NativeError` with a `status` attribute; Python then raises
`RFluxError` with the same status code. The extension uses abi3 wheels on
CPython 3.12+ and version-specific wheels on 3.10/3.11; that policy is separate
from the C ABI promise.
