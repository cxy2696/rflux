/*
 * rflux — stable C ABI for quantitative research primitives.
 *
 * ABI policy
 * ----------
 * - Only this header is guaranteed to be binary-stable across 0.x releases
 *   with the same RFLUX_ABI_VERSION.
 * - Handles are opaque.  Never allocate or inspect them yourself.
 * - All functions return rflux_status.  No C++ exception crosses this boundary.
 * - Output arrays are caller owned.  Their required lengths are documented at
 *   each declaration.  Unless explicitly marked optional, pointers must be
 *   non-NULL.
 * - All functions are deterministic for a given input and seed, except when
 *   explicitly documented otherwise.
 * - Distinct output buffers must not alias each other.  Output arrays are
 *   unspecified after a failed call.
 *
 * Missing-value convention
 * ------------------------
 * Indicator functions propagate missing data: a window containing a non-finite
 * value produces NAN.  Warm-up outputs are NAN.  Cross-sectional and
 * descriptive statistics ignore non-finite inputs unless stated otherwise.
 * HMM fitting and graph construction require finite inputs and return
 * RFLUX_ERR_NONFINITE otherwise.
 */
#ifndef RFLUX_RFLUX_H
#define RFLUX_RFLUX_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && !defined(RFLUX_STATIC)
#  if defined(RFLUX_BUILDING_DLL)
#    define RFLUX_API __declspec(dllexport)
#  else
#    define RFLUX_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define RFLUX_API __attribute__((visibility("default")))
#else
#  define RFLUX_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define RFLUX_ABI_VERSION 1u
#define RFLUX_VERSION_MAJOR 0
#define RFLUX_VERSION_MINOR 4
#define RFLUX_VERSION_PATCH 2

typedef enum rflux_status {
    RFLUX_OK = 0,
    RFLUX_ERR_INVALID_ARGUMENT = 1,
    RFLUX_ERR_OUT_OF_MEMORY = 2,
    RFLUX_ERR_NONFINITE = 3,
    RFLUX_ERR_MODEL_NOT_READY = 4,
    RFLUX_ERR_NUMERICAL = 5,
    RFLUX_ERR_INTERNAL = 6
} rflux_status;

/* ------------------------------------------------------------------------- */
/* Library information                                                       */
/* ------------------------------------------------------------------------- */

RFLUX_API uint32_t rflux_abi_version(void);

/* Returns a static string such as "0.4.2". */
RFLUX_API const char* rflux_version_string(void);

/*
 * Returns a static string; never NULL.  Any integer is accepted; values
 * outside the enum return "RFLUX_ERR_UNKNOWN".
 */
RFLUX_API const char* rflux_status_name(int status);

/*
 * Per-thread error message from the most recent call on this thread.
 * The pointer is valid until the next rflux call on the same thread.  On
 * success the string is empty, not NULL.
 */
RFLUX_API const char* rflux_last_error(void);

RFLUX_API void rflux_clear_error(void);

/* ------------------------------------------------------------------------- */
/* qflux: time-series indicators                                             */
/* ------------------------------------------------------------------------- */

/*
 * Simple returns are x[i] / x[i-1] - 1.  Log returns require positive prices.
 * out has length n; out[0] is NAN and invalid neighbours produce NAN.
 */
RFLUX_API rflux_status rflux_returns(const double* x, size_t n,
                                     int32_t log_returns, double* out);

/* Rolling mean.  out has length n; the first window-1 entries are NAN. */
RFLUX_API rflux_status rflux_sma(const double* x, size_t n, size_t window,
                                 double* out);

/* Rolling sum; same layout as rflux_sma. */
RFLUX_API rflux_status rflux_rolling_sum(const double* x, size_t n,
                                         size_t window, double* out);

/*
 * Rolling standard deviation with ddof degrees of freedom (0 <= ddof < window).
 * Same layout as rflux_sma.
 */
RFLUX_API rflux_status rflux_rolling_std(const double* x, size_t n,
                                         size_t window, size_t ddof,
                                         double* out);

RFLUX_API rflux_status rflux_rolling_min(const double* x, size_t n,
                                         size_t window, double* out);
RFLUX_API rflux_status rflux_rolling_max(const double* x, size_t n,
                                         size_t window, double* out);

/*
 * (x - rolling_mean) / rolling_std.  Uses ddof and returns NAN when the
 * rolling dispersion is zero.
 */
RFLUX_API rflux_status rflux_rolling_zscore(const double* x, size_t n,
                                            size_t window, size_t ddof,
                                            double* out);

/*
 * Exponentially weighted moving average with span >= 1 and
 * alpha = 2 / (span + 1).  The recursion is seeded with the first finite value
 * in each contiguous finite run; a non-finite value resets the recursion.
 * out has length n.
 */
RFLUX_API rflux_status rflux_ema(const double* x, size_t n, size_t span,
                                 double* out);

/*
 * Wilder RSI with a rolling warm-up of `period` price changes.  A flat window
 * with zero average gain and loss maps to 50; zero average loss maps to 100.
 * out has length n.
 */
RFLUX_API rflux_status rflux_rsi(const double* x, size_t n, size_t period,
                                 double* out);

/*
 * MACD(fast, slow, signal).  All EMAs use the rflux_ema seeding convention.
 * fast must be smaller than slow.  macd, signal_line and histogram each have
 * length n and may not alias.
 */
RFLUX_API rflux_status rflux_macd(const double* x, size_t n, size_t fast,
                                  size_t slow, size_t signal, double* macd,
                                  double* signal_line, double* histogram);

/*
 * Bollinger bands: SMA(window) and population standard deviation (ddof = 0),
 * scaled by non-negative n_std.  middle, upper and lower each have length n.
 */
RFLUX_API rflux_status rflux_bollinger(const double* x, size_t n,
                                       size_t window, double n_std,
                                       double* middle, double* upper,
                                       double* lower);

/*
 * Wilder ATR over true range.  TR[0] = high[0] - low[0]; later bars use the
 * previous close when available.  The first ATR appears at index period-1.
 * high, low and close have length n; out has length n.
 */
RFLUX_API rflux_status rflux_atr(const double* high, const double* low,
                                 const double* close, size_t n, size_t period,
                                 double* out);

/* Drawdown from the running peak: equity / peak - 1. */
RFLUX_API rflux_status rflux_drawdown(const double* equity, size_t n,
                                      double* out);

/* Rolling Pearson correlation.  Window must be in [2, n]; out has length n. */
RFLUX_API rflux_status rflux_rolling_corr(const double* x, const double* y,
                                          size_t n, size_t window,
                                          double* out);

/*
 * Rolling covariance over complete windows:
 *   Cov_window(x, y) = sum((x - mean_x)(y - mean_y)) / (window - ddof).
 * ddof must be smaller than window.  Missing values invalidate the window.
 */
RFLUX_API rflux_status rflux_rolling_cov(const double* x, const double* y,
                                         size_t n, size_t window, size_t ddof,
                                         double* out);

/*
 * Rolling OLS beta of y on x:
 *   beta_t = Cov_window(x, y) / Var_window(x).
 * out has length n, requires window >= 2, and is NAN when the window has
 * missing values or the benchmark variance is zero.
 */
RFLUX_API rflux_status rflux_rolling_beta(const double* y, const double* x,
                                          size_t n, size_t window, double* out);

/*
 * Rolling OLS slope of y on t = 0..window-1, and the corresponding R^2.
 * Both outputs have length n and require window >= 2.
 */
RFLUX_API rflux_status rflux_rolling_slope(const double* x, size_t n,
                                           size_t window, double* slope);
RFLUX_API rflux_status rflux_rolling_r2(const double* x, size_t n,
                                        size_t window, double* r2);

/* ------------------------------------------------------------------------- */
/* qflux: statistics                                                         */
/* ------------------------------------------------------------------------- */

/*
 * Descriptive statistics over finite values.
 * variance and stddev use ddof = 1.  skewness is the adjusted
 * Fisher-Pearson G1; excess_kurtosis is the unbiased G2 (normal = 0).
 * n_valid receives the number of finite observations.  Moment outputs that
 * need more observations than are available are NAN.
 */
RFLUX_API rflux_status rflux_describe(const double* x, size_t n, double* mean,
                                      double* variance, double* stddev,
                                      double* skewness, double* excess_kurtosis,
                                      size_t* n_valid);

/*
 * Linear-interpolation quantile (Hyndman-Fan type 7) over finite values.
 * q must be in [0, 1].
 */
RFLUX_API rflux_status rflux_quantile(const double* x, size_t n, double q,
                                      double* out);

/* Pairwise-complete Pearson correlation. */
RFLUX_API rflux_status rflux_pearson(const double* x, const double* y, size_t n,
                                     double* out);

/*
 * Pairwise-complete correlation matrix of a nobs x nassets row-major panel.
 * corr_matrix has length nassets*nassets.  Insufficient or degenerate pairs
 * produce NAN, including on the diagonal.
 */
RFLUX_API rflux_status rflux_corr_matrix(const double* panel, size_t n_obs,
                                         size_t n_assets, double* corr_matrix);

/*
 * Annualized Sharpe ratio from arithmetic periodic returns.  risk_free is an
 * annual rate and is de-annualized by periods_per_year.  Returns NAN for fewer
 * than two finite observations or zero dispersion.
 */
RFLUX_API rflux_status rflux_sharpe(const double* returns, size_t n,
                                    double periods_per_year, double risk_free,
                                    double* out);

/*
 * Annualized Sortino ratio.  Downside deviation uses all excess-return
 * observations and a target of zero.  Returns NAN when there is no downside
 * dispersion.
 */
RFLUX_API rflux_status rflux_sortino(const double* returns, size_t n,
                                     double periods_per_year, double risk_free,
                                     double* out);

/*
 * Historical VaR and CVaR at `confidence` (e.g. 0.95).
 * Both values are reported as positive losses: -quantile(1-confidence) and
 * -mean(return <= that quantile).  They may be negative when the tail is a
 * gain.  NAN inputs are ignored; at least one finite observation is required.
 */
RFLUX_API rflux_status rflux_var_cvar(const double* returns, size_t n,
                                      double confidence, double* var,
                                      double* cvar);

/* ------------------------------------------------------------------------- */
/* qflux: cross-sectional ranking and factor preprocessing                   */
/* ------------------------------------------------------------------------- */

/*
 * Average ascending rank (1-based, ties share their mean rank).  NAN inputs
 * stay NAN and are excluded from the ranking.
 */
RFLUX_API rflux_status rflux_rank(const double* values, size_t n, double* out);

/* Midpoint percentile rank (rank - 0.5) / n_valid; NAN inputs stay NAN. */
RFLUX_API rflux_status rflux_percentile_rank(const double* values, size_t n,
                                             double* out);

/*
 * Cross-sectional z-score over finite values.  ddof = 0 gives population
 * dispersion, ddof = 1 the sample dispersion.  Zero dispersion yields NAN.
 */
RFLUX_API rflux_status rflux_zscore(const double* values, size_t n, size_t ddof,
                                    double* out);

/*
 * Clip finite values to their [lower_q, upper_q] empirical quantiles.  NAN
 * values stay NAN.  Requires 0 <= lower_q <= upper_q <= 1.
 */
RFLUX_API rflux_status rflux_winsorize(const double* values, size_t n,
                                       double lower_q, double upper_q,
                                       double* out);

/*
 * Demean finite values within groups.  group[i] must be -1 (excluded) or in
 * [0, group_count).  Excluded entries produce NAN; NAN values stay NAN.
 */
RFLUX_API rflux_status rflux_neutralize(const double* values, size_t n,
                                        const int32_t* groups,
                                        size_t group_count, double* out);

/* ------------------------------------------------------------------------- */
/* regimelab: Gaussian hidden Markov model (diagonal covariance)             */
/* ------------------------------------------------------------------------- */

typedef struct rflux_hmm rflux_hmm;

/*
 * Options layout is frozen for RFLUX_ABI_VERSION 1.  Always initialise it with
 * rflux_hmm_options_init.  Passing NULL to rflux_hmm_fit selects defaults.
 */
typedef struct rflux_hmm_options {
    size_t struct_size;      /* sizeof(rflux_hmm_options) */
    uint32_t max_iter;       /* default 200 */
    uint32_t n_init;         /* default 3 */
    double tol;              /* default 1e-6 */
    double variance_floor;   /* relative floor, default 1e-6 */
    uint64_t seed;           /* default 42 */
    int32_t sort_states;     /* 1 = sort by first-feature mean, default 1 */
    int32_t reserved0;       /* must be zero */
} rflux_hmm_options;

/*
 * Fill options with defaults.  options_size is sizeof(*options) from the
 * caller.  Fails if the buffer is smaller than the ABI-v1 struct.
 */
RFLUX_API rflux_status rflux_hmm_options_init(rflux_hmm_options* options,
                                              size_t options_size);

/*
 * Create an untrained model.  n_states and n_features must be >= 1 and
 * n_states must fit in int32_t.  The returned handle must be destroyed with
 * rflux_hmm_destroy.
 */
RFLUX_API rflux_status rflux_hmm_create(size_t n_states, size_t n_features,
                                        rflux_hmm** out);

RFLUX_API void rflux_hmm_destroy(rflux_hmm* model);

RFLUX_API rflux_status rflux_hmm_dimensions(const rflux_hmm* model,
                                            size_t* n_states,
                                            size_t* n_features);

/*
 * Fit with Baum-Welch (EM) from n_init deterministic k-medoids restarts,
 * each seeded with k-means++-style distance sampling.
 * observations is row-major n_obs x n_features and must be finite.
 * n_obs must be >= 2 and >= n_states.  max_iter, n_init, tol and
 * variance_floor must be positive.  Multiple random starts use `seed`.
 * A model that reaches max_iter without meeting tol is still returned with
 * converged = 0; this is not an error.
 */
RFLUX_API rflux_status rflux_hmm_fit(rflux_hmm* model,
                                     const double* observations, size_t n_obs,
                                     const rflux_hmm_options* options);

/*
 * Training diagnostics.  Returns RFLUX_ERR_MODEL_NOT_READY unless the current
 * parameters came from rflux_hmm_fit (set_parameters does not create fit
 * diagnostics).
 */
RFLUX_API rflux_status rflux_hmm_fit_summary(const rflux_hmm* model,
                                             size_t* iterations,
                                             int32_t* converged,
                                             double* log_likelihood, double* aic,
                                             double* bic);

/*
 * Log-likelihood of observations under the current parameters.  This is valid
 * after either fit or set_parameters and can score out-of-sample data.
 * It does not require fit diagnostics.
 */
RFLUX_API rflux_status rflux_hmm_score(const rflux_hmm* model,
                                       const double* observations, size_t n_obs,
                                       double* log_likelihood);

/*
 * Parameter buffers have lengths n_states, n_states*n_states, n_states*n_features
 * and n_states*n_features respectively: start probabilities, row-major
 * transition matrix, means, and diagonal variances.  The model must have been
 * fitted or parameterised with rflux_hmm_set_parameters first; otherwise this
 * returns RFLUX_ERR_MODEL_NOT_READY.
 */
RFLUX_API rflux_status rflux_hmm_get_parameters(const rflux_hmm* model,
                                                double* start_prob,
                                                double* transition,
                                                double* means,
                                                double* variances);

/*
 * Install parameters, e.g. loaded from disk.  Makes the model ready for
 * inference; training diagnostics are invalidated.  Values are validated but
 * not re-floored, so exact structural zeros are preserved.
 */
RFLUX_API rflux_status rflux_hmm_set_parameters(rflux_hmm* model,
                                                const double* start_prob,
                                                const double* transition,
                                                const double* means,
                                                const double* variances);

/*
 * Most likely state path (Viterbi, full-sample / non-causal).  states has
 * length n_obs and receives zero-based int32 state ids.
 */
RFLUX_API rflux_status rflux_hmm_viterbi(const rflux_hmm* model,
                                         const double* observations,
                                         size_t n_obs, int32_t* states);

/*
 * Causal filtered state probabilities P(s_t | x_0..x_t).  probs is row-major
 * n_obs x n_states.
 */
RFLUX_API rflux_status rflux_hmm_filtered(const rflux_hmm* model,
                                          const double* observations,
                                          size_t n_obs, double* probs);

/*
 * Full-sample smoothed probabilities P(s_t | x_0..x_{T-1}).  probs is
 * row-major n_obs x n_states.  This is not a trading signal by itself.
 */
RFLUX_API rflux_status rflux_hmm_smoothed(const rflux_hmm* model,
                                          const double* observations,
                                          size_t n_obs, double* probs);

/*
 * Forecast the hidden-state distribution h steps ahead from a current
 * distribution: out = state_probs * transition^h.  out has length n_states.
 * state_probs must be finite, non-negative and have positive mass.
 */
RFLUX_API rflux_status rflux_hmm_forecast(const rflux_hmm* model,
                                          const double* state_probs,
                                          size_t horizon, double* out);

/*
 * One causal HMM update: posterior = normalise(prior * transition * emission),
 * where prior has length n_states and observation has length n_features.
 * out has length n_states.  The prior must be finite, non-negative and have
 * positive total mass.
 */
RFLUX_API rflux_status rflux_hmm_filter_step(
    const rflux_hmm* model, const double* state_probs,
    const double* observation, double* out);

/*
 * Stationary distribution of the current transition matrix.  out has length
 * n_states and sums to one.  Fails with RFLUX_ERR_NUMERICAL when the chain has
 * no unique stationary distribution.
 */
RFLUX_API rflux_status rflux_hmm_stationary(const rflux_hmm* model, double* out);

/*
 * Expected sojourn length of each state: 1 / (1 - A_kk).  Absorbing states
 * produce positive infinity.  out has length n_states.
 */
RFLUX_API rflux_status rflux_hmm_expected_durations(const rflux_hmm* model,
                                                    double* out);

/* ------------------------------------------------------------------------- */
/* regimelab: causal rule-based regimes                                      */
/* ------------------------------------------------------------------------- */

/*
 * Volatility regimes from trailing realized volatility.
 * volatility[t] = sample standard deviation of the last vol_window returns
 * (NAN before the first full window).  vol_window must be at least 2.
 * The current volatility is compared with
 * quantiles of the preceding `lookback` valid volatility observations,
 * excluding the current one.  Labels are 0 = low, 1 = normal, 2 = high and
 * -1 = warm-up/undefined.  At least 20 prior volatility observations are
 * required; lookback must be >= 20.  This rule is strictly causal.
 */
RFLUX_API rflux_status rflux_volatility_regime(const double* returns, size_t n,
                                               size_t vol_window,
                                               size_t lookback, double low_q,
                                               double high_q, int32_t* labels,
                                               double* volatility);

/*
 * Trend regimes from two SMAs of prices.  spread = (fast_ma - slow_ma) /
 * abs(slow_ma).  Labels are 0 = down (spread < -threshold), 1 = sideways,
 * 2 = up (spread > threshold), -1 = warm-up.  fast_ma and slow_ma have length
 * n and are NAN during warm-up.
 */
RFLUX_API rflux_status rflux_trend_regime(const double* prices, size_t n,
                                          size_t fast_window, size_t slow_window,
                                          double threshold, int32_t* labels,
                                          double* fast_ma, double* slow_ma);

/*
 * Combine volatility and trend labels into a risk score:
 * max(volatility_label, 2 - trend_label).  Output is 0 = risk-on,
 * 1 = neutral, 2 = risk-off, -1 = undefined.  Input labels must be in [-1, 2].
 */
RFLUX_API rflux_status rflux_composite_regime(const int32_t* vol_labels,
                                              const int32_t* trend_labels,
                                              size_t n, int32_t* out);

/* ------------------------------------------------------------------------- */
/* percolab: graph and percolation primitives                                */
/* ------------------------------------------------------------------------- */

/* dist[i,j] = sqrt(2 * (1 - corr[i,j])) for i != j; the diagonal is ignored
 * and written as zero.  corr must be a finite symmetric n x n matrix with
 * off-diagonal entries in [-1, 1].  out has length n*n. */
RFLUX_API rflux_status rflux_corr_to_distance(const double* corr, size_t n,
                                              double* out);

/*
 * Connected components of the undirected graph with an edge (i,j) whenever
 * weights[i*n+j] >= threshold and i != j.  The input must be a finite
 * symmetric matrix.  labels has length n; components are numbered 0..C-1 in
 * order of their smallest node.  n_components and largest_size are outputs.
 * n must fit in int32_t because labels are int32_t.
 */
RFLUX_API rflux_status rflux_graph_components(const double* weights, size_t n,
                                              double threshold, int32_t* labels,
                                              size_t* n_components,
                                              size_t* largest_size);

/*
 * Weighted degree / strength for a symmetric non-negative adjacency matrix:
 * out[i] = sum of off-diagonal weights (zeros mean no edge).
 */
RFLUX_API rflux_status rflux_network_strength(const double* weights, size_t n,
                                              double* out);

typedef enum rflux_centrality_kind {
    RFLUX_CENTRALITY_DEGREE = 0,
    RFLUX_CENTRALITY_EIGENVECTOR = 1
} rflux_centrality_kind;

/*
 * Node centrality.  Eigenvector centrality uses L2-normalised power iteration
 * on the off-diagonal adjacency plus an identity shift (A + I), which avoids
 * period-2 oscillation on bipartite graphs.  An all-zero graph yields zeros;
 * failure to converge returns RFLUX_ERR_NUMERICAL.
 */
RFLUX_API rflux_status rflux_network_centrality(const double* weights, size_t n,
                                                uint32_t kind, double* out);

/*
 * Local (unweighted) clustering coefficient of each node.  Nodes with degree
 * below 2 receive NAN.
 */
RFLUX_API rflux_status rflux_network_clustering(const double* weights, size_t n,
                                                double* out);

/*
 * Brandes betweenness centrality for an undirected unweighted graph.  Edges
 * exist where weights[i,j] > 0 and i != j.  When normalized is non-zero,
 * values are divided by the maximum for an undirected graph of this size.
 */
RFLUX_API rflux_status rflux_network_betweenness(
    const double* weights, size_t n, int32_t normalized, double* out);

/*
 * PageRank on a directed non-negative weighted graph.  Rows are outgoing
 * edges; dangling rows distribute their mass uniformly.  damping must be in
 * [0, 1), max_iter and tol must be positive.  out has length n and sums to 1.
 */
RFLUX_API rflux_status rflux_pagerank(const double* adjacency, size_t n,
                                      double damping, size_t max_iter,
                                      double tol, double* out);

/*
 * Minimum spanning tree of a complete symmetric distance matrix.  Edges are
 * emitted in Kruskal order; their buffers must hold at least n-1 entries and
 * may be NULL when n <= 1.  n_edges receives n-1.
 */
RFLUX_API rflux_status rflux_network_mst(const double* distance, size_t n,
                                         uint32_t* edge_u, uint32_t* edge_v,
                                         double* edge_weight,
                                         size_t* n_edges);

/*
 * Classical bond percolation on the graph defined by adjacency > 0.
 * Each candidate edge is retained independently with probability p, then
 * components are recomputed over `repetitions` Monte Carlo draws.
 * probabilities has n_probabilities entries in [0,1].
 *
 * For each p:
 *   giant_mean[p]  = mean(largest component size / n)
 *   giant_std[p]   = population standard deviation of that fraction
 *   susceptibility[p] = mean over draws of
 *                       (sum of s^2 over all non-largest components) / n
 * The finite-cluster susceptibility excludes the largest cluster by design.
 */
RFLUX_API rflux_status rflux_bond_percolation(const double* adjacency, size_t n,
                                              const double* probabilities,
                                              size_t n_probabilities,
                                              size_t repetitions, uint64_t seed,
                                              double* giant_mean,
                                              double* giant_std,
                                              double* susceptibility);

/* Site percolation: each node survives independently with probability p and
 * an edge exists only when both endpoints survive and adjacency > 0.  Giant
 * fractions are relative to the original n.  Outputs follow the bond layout. */
RFLUX_API rflux_status rflux_site_percolation(const double* adjacency, size_t n,
                                              const double* probabilities,
                                              size_t n_probabilities,
                                              size_t repetitions, uint64_t seed,
                                              double* giant_mean,
                                              double* giant_std,
                                              double* susceptibility);

/*
 * Weighted bond percolation: every candidate edge is retained with probability
 * min(1, p * adjacency[i,j]).  Weights must be finite and non-negative.
 * Outputs follow the bond layout.
 */
RFLUX_API rflux_status rflux_bond_percolation_weighted(
    const double* adjacency, size_t n, const double* probabilities,
    size_t n_probabilities, size_t repetitions, uint64_t seed,
    double* giant_mean, double* giant_std, double* susceptibility);

/*
 * Weighted site percolation: node i survives with probability
 * min(1, p * node_weights[i]).  node_weights has length n.  Outputs follow the
 * bond layout.
 */
RFLUX_API rflux_status rflux_site_percolation_weighted(
    const double* adjacency, size_t n, const double* node_weights,
    const double* probabilities, size_t n_probabilities, size_t repetitions,
    uint64_t seed, double* giant_mean, double* giant_std,
    double* susceptibility);

/*
 * Stochastic SIR risk diffusion on a symmetric non-negative weighted graph.
 * Infected node i attempts to infect susceptible neighbour j with probability
 * 1 - exp(-beta * w_ij) per step.  Infected nodes recover with probability
 * gamma per step, after transmitting.  Infections and recoveries within a
 * step are simultaneous.
 *
 * mean_affected has length steps+1 and receives the mean fraction (I+R)/n,
 * including the initial step.  infection_frequency has length n and receives
 * the ever-infected probability across repetitions.  Duplicate seed nodes are
 * ignored.  n must fit in uint32_t because seed indices are uint32_t.
 */
RFLUX_API rflux_status rflux_risk_diffusion(const double* adjacency, size_t n,
                                            const uint32_t* seed_nodes,
                                            size_t n_seed_nodes, double beta,
                                            double gamma, size_t steps,
                                            size_t repetitions, uint64_t seed,
                                            double* mean_affected,
                                            double* infection_frequency);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RFLUX_RFLUX_H */
