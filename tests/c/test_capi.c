#include "rflux/rflux.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition)                                                     \
    do {                                                                     \
        if (!(condition)) {                                                  \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,          \
                    #condition);                                             \
            return 1;                                                        \
        }                                                                    \
    } while (0)

static int approx(double actual, double expected, double tolerance) {
    if (isnan(actual) || isnan(expected)) return 0;
    return fabs(actual - expected) <= tolerance;
}

static int giant_at(const double* giant, double first, double second) {
    return approx(giant[0], first, 1e-12) && approx(giant[1], second, 1e-12);
}


/* Graphs and percolation inputs shared by several test groups. */
static const double kComplete4[16] = {
    0, 1, 1, 1,
    1, 0, 1, 1,
    1, 1, 0, 1,
    1, 1, 1, 0};
static const double kProbabilities01[2] = {0.0, 1.0};
static const double kPath4[16] = {
    0, 1, 0, 0,
    1, 0, 1, 0,
    0, 1, 0, 1,
    0, 0, 1, 0};
static const double kLeadNodeWeights[4] = {1.0, 0.0, 0.0, 0.0};

static int two_pair_labels(const int32_t* labels, size_t n_components,
                           size_t largest) {
    return n_components == 2 && largest == 2 && labels[0] == labels[1] &&
           labels[2] == labels[3] && labels[0] != labels[2];
}

static int test_status(void) {
    double out[5];
    const double x[5] = {1, 2, 3, 4, 5};
    CHECK(rflux_abi_version() == RFLUX_ABI_VERSION);
    CHECK(strcmp(rflux_version_string(), "0.7.1") == 0);
    CHECK(strcmp(rflux_status_name(RFLUX_OK), "RFLUX_OK") == 0);
    CHECK(strcmp(rflux_status_name(999), "RFLUX_ERR_UNKNOWN") ==
          0);

    rflux_clear_error();
    CHECK(strlen(rflux_last_error()) == 0);
    CHECK(rflux_sma(NULL, 5, 2, out) == RFLUX_ERR_INVALID_ARGUMENT);
    CHECK(strlen(rflux_last_error()) > 0);
    CHECK(rflux_sma(x, 5, 2, out) == RFLUX_OK);
    CHECK(strlen(rflux_last_error()) == 0);
    CHECK(rflux_sma(x, 5, 2, NULL) == RFLUX_ERR_INVALID_ARGUMENT);
    return 0;
}

static int test_indicators(void) {
    const double x[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double out[16];
    double second[16];
    CHECK(rflux_sma(x, 10, 3, out) == RFLUX_OK);
    CHECK(isnan(out[0]) && isnan(out[1]));
    CHECK(approx(out[2], 2.0, 1e-12));
    CHECK(approx(out[9], 9.0, 1e-12));

    CHECK(rflux_returns(x, 10, 0, out) == RFLUX_OK);
    CHECK(isnan(out[0]));
    CHECK(approx(out[1], 1.0, 1e-12));
    CHECK(rflux_returns(x, 10, 1, out) == RFLUX_OK);
    CHECK(approx(out[1], log(2.0), 1e-12));

    CHECK(rflux_rolling_sum(x, 10, 3, out) == RFLUX_OK);
    CHECK(approx(out[9], 27.0, 1e-12));
    CHECK(rflux_rolling_std(x, 10, 3, 0, out) == RFLUX_OK);
    CHECK(approx(out[2], sqrt(2.0 / 3.0), 1e-12));
    CHECK(rflux_rolling_std(x, 10, 3, 1, out) == RFLUX_OK);
    CHECK(approx(out[2], 1.0, 1e-12));
    CHECK(rflux_rolling_min(x, 10, 3, out) == RFLUX_OK);
    CHECK(approx(out[9], 8.0, 1e-12));
    CHECK(rflux_rolling_max(x, 10, 3, out) == RFLUX_OK);
    CHECK(approx(out[9], 10.0, 1e-12));
    CHECK(rflux_rolling_zscore(x, 10, 2, 0, out) == RFLUX_OK);
    CHECK(approx(out[9], 1.0, 1e-12));

    CHECK(rflux_ema(x, 10, 3, out) == RFLUX_OK);
    CHECK(approx(out[0], 1.0, 1e-12));
    CHECK(approx(out[1], 1.5, 1e-12));
    CHECK(approx(out[2], 2.25, 1e-12));

    {
        double prices[16];
        for (int i = 0; i < 16; ++i) prices[i] = (double)(i + 1);
        CHECK(rflux_rsi(prices, 16, 14, out) == RFLUX_OK);
        CHECK(isnan(out[13]));
        CHECK(approx(out[14], 100.0, 1e-12));
        CHECK(approx(out[15], 100.0, 1e-12));
    }

    {
        double prices[50];
        double macd[50], signal[50], hist[50];
        for (int i = 0; i < 50; ++i) prices[i] = 1.0 + 0.02 * i;
        CHECK(rflux_macd(prices, 50, 12, 26, 9, macd, signal, hist) ==
              RFLUX_OK);
        CHECK(approx(hist[49], macd[49] - signal[49], 1e-12));
        CHECK(rflux_bollinger(x, 10, 10, 2.0, out, second, NULL) ==
              RFLUX_ERR_INVALID_ARGUMENT);
    }

    {
        double prices[20];
        double middle[20], upper[20], lower[20];
        for (int i = 0; i < 20; ++i) prices[i] = (double)(i + 1);
        CHECK(rflux_bollinger(prices, 20, 20, 2.0, middle, upper, lower) ==
              RFLUX_OK);
        CHECK(approx(middle[19], 10.5, 1e-12));
        CHECK(approx(upper[19], 10.5 + 2.0 * sqrt(33.25), 1e-12));
    }

    {
        double high[6], low[6], close[6], atr[6];
        for (int i = 0; i < 6; ++i) {
            close[i] = (double)(i + 1);
            high[i] = close[i] + 1.0;
            low[i] = close[i] - 1.0;
        }
        CHECK(rflux_atr(high, low, close, 6, 3, atr) == RFLUX_OK);
        CHECK(isnan(atr[1]));
        CHECK(approx(atr[2], 2.0, 1e-12));
        CHECK(approx(atr[5], 2.0, 1e-12));
    }

    {
        const double equity[4] = {100.0, 110.0, 99.0, 120.0};
        CHECK(rflux_drawdown(equity, 4, out) == RFLUX_OK);
        CHECK(approx(out[2], -0.1, 1e-12));
        CHECK(approx(out[3], 0.0, 1e-12));
    }

    CHECK(rflux_rolling_corr(x, x, 10, 3, out) == RFLUX_OK);
    CHECK(approx(out[9], 1.0, 1e-12));
    {
        double benchmark[10];
        double asset[10];
        for (int i = 0; i < 10; ++i) {
            benchmark[i] = (double)(i + 1);
            asset[i] = 2.0 * benchmark[i] + 3.0;
        }
        CHECK(rflux_rolling_beta(asset, benchmark, 10, 5, out) == RFLUX_OK);
        CHECK(approx(out[4], 2.0, 1e-12));
        CHECK(approx(out[9], 2.0, 1e-12));
        CHECK(rflux_rolling_cov(benchmark, asset, 10, 5, 0, out) == RFLUX_OK);
        CHECK(approx(out[4], 4.0, 1e-12));
        CHECK(approx(out[9], 4.0, 1e-12));
        CHECK(rflux_rolling_cov(benchmark, asset, 10, 5, 1, out) == RFLUX_OK);
        CHECK(approx(out[4], 5.0, 1e-12));
        CHECK(rflux_rolling_cov(benchmark, asset, 10, 5, 5, out) ==
              RFLUX_ERR_INVALID_ARGUMENT);
    }
    CHECK(rflux_rolling_slope(x, 10, 5, out) == RFLUX_OK);
    CHECK(approx(out[9], 1.0, 1e-12));
    CHECK(rflux_rolling_r2(x, 10, 5, out) == RFLUX_OK);
    CHECK(approx(out[9], 1.0, 1e-12));

    {
        const double missing[6] = {1.0, 2.0, NAN, 4.0, 5.0, 6.0};
        CHECK(rflux_sma(missing, 6, 2, out) == RFLUX_OK);
        CHECK(approx(out[1], 1.5, 1e-12));
        CHECK(isnan(out[2]) && isnan(out[3]));
        CHECK(approx(out[4], 4.5, 1e-12));
    }
    return 0;
}

static int test_stats(void) {
    const double x[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    double mean, variance, sd, skew, kurt, out, var, cvar;
    size_t n_valid;
    CHECK(rflux_describe(x, 10, &mean, &variance, &sd, &skew, &kurt,
                         &n_valid) == RFLUX_OK);
    CHECK(n_valid == 10);
    CHECK(approx(mean, 5.5, 1e-12));
    CHECK(approx(variance, 55.0 / 6.0, 1e-12));
    CHECK(approx(skew, 0.0, 1e-12));
    CHECK(approx(kurt, -1.2, 1e-12));
    CHECK(rflux_quantile(x, 10, 0.25, &out) == RFLUX_OK);
    CHECK(approx(out, 3.25, 1e-12));

    {
        const double y[5] = {2, 4, 6, 8, 10};
        double short_x[5] = {1, 2, 3, 4, 5};
        CHECK(rflux_pearson(short_x, y, 5, &out) == RFLUX_OK);
        CHECK(approx(out, 1.0, 1e-12));
    }

    {
        const double panel[15] = {
            1, 2, 5,
            2, 4, 4,
            3, NAN, 3,
            NAN, 8, 2,
            5, 10, 1};
        double corr[9];
        CHECK(rflux_corr_matrix(panel, 5, 3, corr) == RFLUX_OK);
        CHECK(approx(corr[0], 1.0, 1e-12));
        CHECK(approx(corr[1], 1.0, 1e-12));
        CHECK(approx(corr[3], 1.0, 1e-12));
        CHECK(approx(corr[4], 1.0, 1e-12));
        CHECK(approx(corr[8], 1.0, 1e-12));
        CHECK(approx(corr[1], corr[3], 1e-12));
    }

    {
        const double returns[2] = {0.02, -0.01};
        CHECK(rflux_sharpe(returns, 2, 252.0, 0.0, &out) == RFLUX_OK);
        CHECK(approx(out, 3.7416573868, 1e-8));
        CHECK(rflux_sortino(returns, 2, 252.0, 0.0, &out) == RFLUX_OK);
        CHECK(approx(out, 11.2249721603, 1e-8));
    }

    {
        const double returns[5] = {-0.1, -0.05, 0.0, 0.05, 0.1};
        CHECK(rflux_var_cvar(returns, 5, 0.8, &var, &cvar) == RFLUX_OK);
        CHECK(approx(var, 0.06, 1e-12));
        CHECK(approx(cvar, 0.1, 1e-12));
    }
    return 0;
}

static int test_ranking(void) {
    const double values[5] = {3, 1, 1, NAN, 5};
    double out[10];
    CHECK(rflux_rank(values, 5, out) == RFLUX_OK);
    CHECK(approx(out[0], 3.0, 1e-12));
    CHECK(approx(out[1], 1.5, 1e-12));
    CHECK(approx(out[2], 1.5, 1e-12));
    CHECK(isnan(out[3]));
    CHECK(approx(out[4], 4.0, 1e-12));

    CHECK(rflux_percentile_rank(values, 5, out) == RFLUX_OK);
    CHECK(approx(out[0], 2.5 / 4.0, 1e-12));

    {
        const double sample[3] = {1, 2, 3};
        CHECK(rflux_zscore(sample, 3, 0, out) == RFLUX_OK);
        CHECK(approx(out[0], -1.22474487139, 1e-9));
        CHECK(approx(out[1], 0.0, 1e-12));
        CHECK(rflux_rolling_quantile(sample, 3, 3, 0.5, out) == RFLUX_OK);
        CHECK(approx(out[2], 2.0, 1e-12));
        CHECK(rflux_rolling_quantile(sample, 3, 3, 1.1, out) ==
              RFLUX_ERR_INVALID_ARGUMENT);
    }

    {
        const double sample[10] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        CHECK(rflux_winsorize(sample, 10, 0.1, 0.9, out) == RFLUX_OK);
        CHECK(approx(out[0], 1.9, 1e-12));
        CHECK(approx(out[9], 9.1, 1e-12));
    }

    {
        const double sample[4] = {1, 2, 10, 12};
        const int32_t groups[4] = {0, 0, 1, 1};
        CHECK(rflux_neutralize(sample, 4, groups, 2, out) == RFLUX_OK);
        CHECK(approx(out[0], -0.5, 1e-12));
        CHECK(approx(out[1], 0.5, 1e-12));
        CHECK(approx(out[2], -1.0, 1e-12));
        CHECK(approx(out[3], 1.0, 1e-12));
    }
    return 0;
}

static int test_hmm(void) {
    const size_t n = 600;
    double* observations = (double*)malloc(n * sizeof(double));
    int32_t* path = (int32_t*)malloc(n * sizeof(int32_t));
    CHECK(observations != NULL && path != NULL);
    for (size_t i = 0; i < n; ++i) {
        observations[i] = (i < n / 2 ? -1.0 : 1.0) +
                          0.15 * sin((double)i * 0.37);
    }

    rflux_hmm* model = NULL;
    CHECK(rflux_hmm_create(2, 1, &model) == RFLUX_OK);
    rflux_hmm* untouched = NULL;
    CHECK(rflux_hmm_create(2, 1, &untouched) == RFLUX_OK);
    {
        size_t iterations;
        int32_t converged;
        double ll, aic, bic;
        CHECK(rflux_hmm_fit_summary(untouched, &iterations, &converged, &ll,
                                    &aic, &bic) == RFLUX_ERR_MODEL_NOT_READY);
        double start[2], transition[4], means[2], variances[2];
        CHECK(rflux_hmm_get_parameters(untouched, start, transition, means,
                                       variances) == RFLUX_ERR_MODEL_NOT_READY);
    }

    {
        double start[2] = {1.0, 0.0};
        double transition[4] = {1.0, 0.0, 0.0, 1.0};
        double means[2] = {-1.0, 1.0};
        double variances[2] = {1.0, 1.0};
        double check_start[2], check_transition[4], check_means[2];
        double check_variances[2], forecast[2];
        rflux_hmm* structured = NULL;
        CHECK(rflux_hmm_create(2, 1, &structured) == RFLUX_OK);
        CHECK(rflux_hmm_set_parameters(structured, start, transition, means,
                                       variances) == RFLUX_OK);
        CHECK(rflux_hmm_get_parameters(structured, check_start,
                                       check_transition, check_means,
                                       check_variances) == RFLUX_OK);
        CHECK(memcmp(start, check_start, sizeof(start)) == 0);
        CHECK(memcmp(transition, check_transition, sizeof(transition)) == 0);
        CHECK(rflux_hmm_forecast(structured, start, 5, forecast) == RFLUX_OK);
        CHECK(forecast[0] == 1.0 && forecast[1] == 0.0);
        rflux_hmm_destroy(structured);
    }

    rflux_hmm_options options;
    CHECK(rflux_hmm_options_init(&options, sizeof(options)) == RFLUX_OK);
    CHECK(options.struct_size == sizeof(options));
    options.max_iter = 250;
    options.n_init = 3;
    options.seed = 7;
    CHECK(rflux_hmm_fit(model, observations, n, &options) == RFLUX_OK);

    {
        size_t iterations, states, features;
        int32_t converged;
        double ll, aic, bic;
        CHECK(rflux_hmm_dimensions(model, &states, &features) == RFLUX_OK);
        CHECK(states == 2 && features == 1);
        CHECK(rflux_hmm_fit_summary(model, &iterations, &converged, &ll, &aic,
                                    &bic) == RFLUX_OK);
        CHECK(iterations > 0);
        CHECK(isfinite(ll) && isfinite(aic) && isfinite(bic));
        {
            double score = 0.0;
            CHECK(rflux_hmm_score(model, observations, n, &score) == RFLUX_OK);
            CHECK(approx(score, ll, 1e-8));
        }
    }

    {
        double start[2], transition[4], means[2], variances[2];
        double start_sum = 0.0;
        CHECK(rflux_hmm_get_parameters(model, start, transition, means,
                                       variances) == RFLUX_OK);
        for (size_t i = 0; i < 2; ++i) start_sum += start[i];
        CHECK(approx(start_sum, 1.0, 1e-9));
        for (size_t row = 0; row < 2; ++row) {
            CHECK(approx(transition[row * 2] + transition[row * 2 + 1], 1.0,
                         1e-9));
        }
        CHECK(means[0] < means[1]);
        CHECK(variances[0] > 0.0 && variances[1] > 0.0);
    }

    CHECK(rflux_hmm_viterbi(model, observations, n, path) == RFLUX_OK);
    {
        size_t first_zero = 0;
        size_t last_one = 0;
        for (size_t i = 0; i < n / 2; ++i) first_zero += (path[i] == 0);
        for (size_t i = n / 2; i < n; ++i) last_one += (path[i] == 1);
        CHECK(first_zero > 290);
        CHECK(last_one > 290);
    }

    {
        const size_t prefix = 200;
        double* full = (double*)malloc(n * 2 * sizeof(double));
        double* partial = (double*)malloc(prefix * 2 * sizeof(double));
        CHECK(full != NULL && partial != NULL);
        CHECK(rflux_hmm_filtered(model, observations, n, full) == RFLUX_OK);
        CHECK(rflux_hmm_filtered(model, observations, prefix, partial) ==
              RFLUX_OK);
        for (size_t i = 0; i < prefix * 2; ++i) {
            CHECK(approx(full[i], partial[i], 1e-11));
        }
        for (size_t t = 0; t < n; ++t) {
            CHECK(approx(full[t * 2] + full[t * 2 + 1], 1.0, 1e-9));
        }
        free(full);
        free(partial);
    }

    {
        double current[2] = {1.0, 0.0};
        double forecast[2];
        CHECK(rflux_hmm_forecast(model, current, 10, forecast) == RFLUX_OK);
        CHECK(approx(forecast[0] + forecast[1], 1.0, 1e-9));
        CHECK(forecast[0] >= 0.0 && forecast[1] >= 0.0);
        double observation = 0.0;
        double step[2];
        double prior[2] = {0.6, 0.4};
        CHECK(rflux_hmm_filter_step(model, prior, &observation, step) == RFLUX_OK);
        CHECK(approx(step[0] + step[1], 1.0, 1e-9));
        double stationary[2];
        CHECK(rflux_hmm_stationary(model, stationary) == RFLUX_OK);
        CHECK(approx(stationary[0] + stationary[1], 1.0, 1e-9));
        double durations[2];
        CHECK(rflux_hmm_expected_durations(model, durations) == RFLUX_OK);
        CHECK(durations[0] > 0.0 && durations[1] > 0.0);
    }

    {
        double start[2], transition[4], means[2], variances[2];
        int32_t* copy_path = (int32_t*)malloc(n * sizeof(int32_t));
        CHECK(copy_path != NULL);
        CHECK(rflux_hmm_get_parameters(model, start, transition, means,
                                       variances) == RFLUX_OK);
        rflux_hmm* copy = NULL;
        CHECK(rflux_hmm_create(2, 1, &copy) == RFLUX_OK);
        CHECK(rflux_hmm_set_parameters(copy, start, transition, means,
                                       variances) == RFLUX_OK);
        CHECK(rflux_hmm_viterbi(copy, observations, n, copy_path) == RFLUX_OK);
        CHECK(memcmp(path, copy_path, n * sizeof(int32_t)) == 0);
        free(copy_path);
        rflux_hmm_destroy(copy);
    }

    rflux_hmm_destroy(untouched);
    rflux_hmm_destroy(model);
    free(observations);
    free(path);
    return 0;
}

static int test_regimes(void) {
    const size_t n = 400;
    double* returns = (double*)malloc(n * sizeof(double));
    double* prices = (double*)malloc(n * sizeof(double));
    int32_t* labels = (int32_t*)malloc(n * sizeof(int32_t));
    int32_t* trend_labels = (int32_t*)malloc(n * sizeof(int32_t));
    int32_t* composite = (int32_t*)malloc(n * sizeof(int32_t));
    double* volatility = (double*)malloc(n * sizeof(double));
    double* fast = (double*)malloc(n * sizeof(double));
    double* slow = (double*)malloc(n * sizeof(double));
    CHECK(returns && prices && labels && trend_labels && composite &&
          volatility && fast && slow);
    prices[0] = 100.0;
    returns[0] = NAN;
    for (size_t i = 1; i < n; ++i) {
        returns[i] = (i < 200 ? 0.005 : 0.02) * sin((double)i * 0.37);
        prices[i] = 100.0 + 0.2 * (double)i + sin((double)i * 0.1);
    }
    CHECK(rflux_volatility_regime(returns, n, 20, 60, 0.25, 0.75, labels,
                                  volatility) == RFLUX_OK);
    CHECK(labels[0] == -1);
    {
        int valid = 0;
        for (size_t i = 0; i < n; ++i) {
            CHECK(labels[i] >= -1 && labels[i] <= 2);
            if (labels[i] >= 0) ++valid;
        }
        CHECK(valid > 0);
    }

    CHECK(rflux_trend_regime(prices, n, 20, 60, 0.0, trend_labels, fast,
                             slow) == RFLUX_OK);
    CHECK(trend_labels[n - 1] == 2);
    CHECK(isfinite(fast[n - 1]) && isfinite(slow[n - 1]));
    CHECK(fast[n - 1] > slow[n - 1]);
    CHECK(rflux_composite_regime(labels, trend_labels, n, composite) ==
          RFLUX_OK);
    for (size_t i = 0; i < n; ++i) {
        if (labels[i] < 0 || trend_labels[i] < 0) {
            CHECK(composite[i] == -1);
        } else {
            int expected = labels[i] > 2 - trend_labels[i]
                               ? labels[i]
                               : 2 - trend_labels[i];
            CHECK(composite[i] == expected);
        }
    }

    free(returns);
    free(prices);
    free(labels);
    free(trend_labels);
    free(composite);
    free(volatility);
    free(fast);
    free(slow);
    return 0;
}

static int test_graphs(void) {
    const double corr[16] = {
        1.0, 0.9, 0.2, 0.1,
        0.9, 1.0, 0.3, 0.1,
        0.2, 0.3, 1.0, 0.8,
        0.1, 0.1, 0.8, 1.0};
    double distance[16];
    double strength[4];
    double eigen[4];
    int32_t labels[4];
    size_t n_components = 0;
    size_t largest = 0;
    size_t n_edges = 0;

    CHECK(rflux_corr_to_distance(corr, 4, distance) == RFLUX_OK);
    CHECK(approx(distance[1], sqrt(2.0 * 0.1), 1e-12));
    {
        const double diagonal_free[4] = {0.0, 0.5, 0.5, 0.0};
        double distance_2[4];
        CHECK(rflux_corr_to_distance(diagonal_free, 2, distance_2) ==
              RFLUX_OK);
        CHECK(distance_2[0] == 0.0 && distance_2[3] == 0.0);
        CHECK(approx(distance_2[1], 1.0, 1e-12));
        CHECK(approx(distance_2[2], 1.0, 1e-12));
    }
    CHECK(rflux_graph_components(corr, 4, 0.5, labels, &n_components,
                                 &largest) == RFLUX_OK);
    CHECK(two_pair_labels(labels, n_components, largest));

    CHECK(rflux_network_strength(corr, 4, strength) == RFLUX_OK);
    CHECK(approx(strength[0], 1.2, 1e-12));
    CHECK(rflux_network_centrality(corr, 4, RFLUX_CENTRALITY_EIGENVECTOR,
                                   eigen) == RFLUX_OK);
    {
        double norm = 0.0;
        for (size_t i = 0; i < 4; ++i) norm += eigen[i] * eigen[i];
        CHECK(approx(norm, 1.0, 1e-9));
    }
    {
        const double path[9] = {
            0, 1, 0,
            1, 0, 1,
            0, 1, 0};
        double path_cluster[3];
        CHECK(rflux_network_clustering(path, 3, path_cluster) == RFLUX_OK);
        CHECK(isnan(path_cluster[0]));
        CHECK(approx(path_cluster[1], 0.0, 1e-12));
        CHECK(isnan(path_cluster[2]));

        double path_centrality[3];
        CHECK(rflux_network_centrality(path, 3,
                                       RFLUX_CENTRALITY_EIGENVECTOR,
                                       path_centrality) == RFLUX_OK);
        CHECK(approx(path_centrality[0], 0.5, 1e-8));
        CHECK(approx(path_centrality[1], sqrt(0.5), 1e-8));
        CHECK(approx(path_centrality[2], 0.5, 1e-8));
    }

    {
        double efficiency = 0.0;
        double average_path_length = 0.0;
        size_t diameter = 0;
        size_t reachable = 0;
        int32_t connected = 0;
        CHECK(rflux_network_path_stats(kPath4, 4, &efficiency,
                                       &average_path_length, &diameter,
                                       &reachable, &connected) == RFLUX_OK);
        CHECK(approx(efficiency, 13.0 / 18.0, 1e-12));
        CHECK(approx(average_path_length, 5.0 / 3.0, 1e-12));
        CHECK(diameter == 3 && reachable == 6 && connected == 1);

        int32_t cores[4];
        CHECK(rflux_network_core_numbers(kPath4, 4, cores) == RFLUX_OK);
        for (size_t i = 0; i < 4; ++i) CHECK(cores[i] == 1);

        double global_clustering = 0.0;
        CHECK(rflux_network_transitivity(kPath4, 4, &global_clustering) ==
              RFLUX_OK);
        CHECK(approx(global_clustering, 0.0, 1e-12));
        double assortativity = 0.0;
        CHECK(rflux_network_assortativity(kPath4, 4, &assortativity) ==
              RFLUX_OK);
        CHECK(approx(assortativity, -0.5, 1e-12));
        const int32_t partition[4] = {0, 0, 1, 1};
        double modularity = 0.0;
        CHECK(rflux_network_modularity(kPath4, 4, partition, 2, &modularity) ==
              RFLUX_OK);
        CHECK(approx(modularity, 1.0 / 6.0, 1e-12));
        CHECK(rflux_network_modularity(kPath4, 4, partition, 5, &modularity) ==
              RFLUX_ERR_INVALID_ARGUMENT);
        CHECK(rflux_network_path_stats(kPath4, 4, NULL, &average_path_length,
                                       &diameter, &reachable, &connected) ==
              RFLUX_ERR_INVALID_ARGUMENT);

        const double distances[16] = {
            0, 1, 3, 4,
            1, 0, 2, 5,
            3, 2, 0, 1,
            4, 5, 1, 0};
        CHECK(rflux_mst_clusters(distances, 4, 1.5, labels, &n_components,
                                 &largest) == RFLUX_OK);
        CHECK(two_pair_labels(labels, n_components, largest));
    }

    {
        uint32_t bridge_u[3];
        uint32_t bridge_v[3];
        size_t n_bridges = 0;
        CHECK(rflux_network_bridges(kPath4, 4, bridge_u, bridge_v,
                                    &n_bridges) == RFLUX_OK);
        CHECK(n_bridges == 3);
        CHECK(bridge_u[0] == 0 && bridge_v[0] == 1);
        CHECK(bridge_u[1] == 1 && bridge_v[1] == 2);
        CHECK(bridge_u[2] == 2 && bridge_v[2] == 3);
        CHECK(rflux_network_bridges(kPath4, 4, NULL, NULL, &n_bridges) ==
              RFLUX_ERR_INVALID_ARGUMENT);

        int32_t art[4];
        CHECK(rflux_network_articulation_points(kPath4, 4, art) == RFLUX_OK);
        CHECK(art[0] == 0 && art[1] == 1 && art[2] == 1 && art[3] == 0);

        int32_t truss[16];
        CHECK(rflux_network_truss_numbers(kPath4, 4, truss) == RFLUX_OK);
        CHECK(truss[1] == 2 && truss[4 * 2 + 3] == 2);
        CHECK(truss[0] == 0 && truss[2] == 0);

        const double single[1] = {0.0};
        CHECK(rflux_network_bridges(single, 1, NULL, NULL, &n_bridges) ==
              RFLUX_OK);
        CHECK(n_bridges == 0);
    }

    {
        uint32_t u[3], v[3];
        double w[3];
        CHECK(rflux_network_mst(distance, 4, u, v, w, &n_edges) == RFLUX_OK);
        CHECK(n_edges == 3);
        for (size_t i = 0; i < n_edges; ++i) {
            CHECK(u[i] < 4 && v[i] < 4 && u[i] != v[i]);
            CHECK(isfinite(w[i]) && w[i] >= 0.0);
        }
    }

    {
        const double cycle[9] = {
            0, 1, 0,
            0, 0, 1,
            1, 0, 0};
        double ranks[3];
        CHECK(rflux_pagerank(cycle, 3, 0.85, 100, 1e-12, ranks) == RFLUX_OK);
        CHECK(approx(ranks[0], 1.0 / 3.0, 1e-10));
        CHECK(approx(ranks[1], 1.0 / 3.0, 1e-10));
        CHECK(approx(ranks[2], 1.0 / 3.0, 1e-10));
        CHECK(rflux_pagerank(cycle, 3, 1.0, 100, 1e-12, ranks) ==
              RFLUX_ERR_INVALID_ARGUMENT);
    }

    {
        double giant[2], giant_std[2], susceptibility[2];
        CHECK(rflux_bond_percolation(kComplete4, 4, kProbabilities01, 2, 20, 3,
                                     giant, giant_std, susceptibility) ==
              RFLUX_OK);
        CHECK(giant_at(giant, 0.25, 1.0));
        CHECK(approx(giant_std[0], 0.0, 1e-6));
        CHECK(approx(susceptibility[0], 0.75, 1e-12));
        CHECK(approx(susceptibility[1], 0.0, 1e-12));

        CHECK(rflux_site_percolation(kComplete4, 4, kProbabilities01, 2, 20, 3,
                                     giant, giant_std, susceptibility) ==
              RFLUX_OK);
        CHECK(giant_at(giant, 0.0, 1.0));
    }

    {
        const double path3[9] = {
            0, 1, 0,
            1, 0, 1,
            0, 1, 0};
        double between[3];
        CHECK(rflux_network_betweenness(path3, 3, 1, between) == RFLUX_OK);
        CHECK(approx(between[1], 1.0, 1e-12));
        CHECK(approx(between[0], 0.0, 1e-12));
        CHECK(rflux_network_betweenness(path3, 3, 0, between) == RFLUX_OK);
        CHECK(approx(between[1], 1.0, 1e-12));
    }

    {
        double giant[2], giant_std[2], susceptibility[2];
        CHECK(rflux_bond_percolation_weighted(
                  kComplete4, 4, kProbabilities01, 2, 20, 3, giant, giant_std,
                  susceptibility) == RFLUX_OK);
        CHECK(giant_at(giant, 0.25, 1.0));
        CHECK(rflux_site_percolation_weighted(
                  kComplete4, 4, kLeadNodeWeights, kProbabilities01, 2, 20, 3,
                  giant, giant_std, susceptibility) == RFLUX_OK);
        CHECK(giant_at(giant, 0.0, 0.25));
    }

    {
        double giant[2], giant_std[2], susceptibility[2];
        CHECK(rflux_percolation_scan(kComplete4, 4, NULL, kLeadNodeWeights, 1,
                                     kProbabilities01, 2, 20, 3, giant, giant_std,
                                     susceptibility) == RFLUX_OK);
        CHECK(giant_at(giant, 0.0, 0.25));
        CHECK(rflux_percolation_scan(kComplete4, 4, NULL, kLeadNodeWeights, 2,
                                     kProbabilities01, 2, 20, 3, giant, giant_std,
                                     susceptibility) ==
              RFLUX_ERR_INVALID_ARGUMENT);
        CHECK(rflux_percolation_scan(kComplete4, 4, NULL, kLeadNodeWeights, 0,
                                     kProbabilities01, 2, 20, 3, giant, giant_std,
                                     susceptibility) ==
              RFLUX_ERR_INVALID_ARGUMENT);

        double distribution[5];
        CHECK(rflux_cluster_size_distribution(kComplete4, 4, NULL, NULL, 0, 0.0,
                                              5, 7, distribution) == RFLUX_OK);
        CHECK(distribution[0] == 0.0);
        CHECK(approx(distribution[1], 4.0, 1e-12));
        CHECK(approx(distribution[4], 0.0, 1e-12));
        CHECK(rflux_cluster_size_distribution(kComplete4, 4, NULL, NULL, 0, 1.0,
                                              5, 7, distribution) == RFLUX_OK);
        CHECK(approx(distribution[4], 1.0, 1e-12));

        const uint32_t order[4] = {0, 1, 2, 3};
        size_t removed[5];
        double giant_curve[5];
        double susceptibility_curve[5];
        size_t components[5];
        CHECK(rflux_targeted_removal(kComplete4, 4, order, 4, removed,
                                     giant_curve, susceptibility_curve,
                                     components) == RFLUX_OK);
        CHECK(removed[0] == 0 && removed[4] == 4);
        CHECK(approx(giant_curve[0], 1.0, 1e-12));
        CHECK(approx(giant_curve[4], 0.0, 1e-12));
        CHECK(components[0] == 1 && components[4] == 0);
        CHECK(rflux_targeted_removal(kComplete4, 4, order, 0, removed,
                                     giant_curve, susceptibility_curve,
                                     components) ==
              RFLUX_ERR_INVALID_ARGUMENT);
    }

    {
        const uint32_t seeds[1] = {0};
        double curve[4];
        double frequency[4];
        CHECK(rflux_risk_diffusion(kComplete4, 4, seeds, 1, 10.0, 0.0, 3, 5, 9,
                                   curve, frequency) == RFLUX_OK);
        CHECK(approx(curve[0], 0.25, 1e-12));
        CHECK(curve[1] > 0.9);
        CHECK(approx(frequency[0], 1.0, 1e-12));
        for (size_t i = 0; i < 4; ++i) {
            CHECK(frequency[i] >= 0.0 && frequency[i] <= 1.0);
        }
        CHECK(rflux_risk_diffusion(kComplete4, 4, seeds, 1, 10.0, 0.0,
                                   SIZE_MAX, 5, 9, curve, frequency) ==
              RFLUX_ERR_INVALID_ARGUMENT);
    }

    {
        double bad[4] = {1, 0, 0, 0};
        bad[1] = NAN;
        CHECK(rflux_graph_components(bad, 2, 0.0, labels, &n_components,
                                     &largest) == RFLUX_ERR_NONFINITE);
    }
    return 0;
}

static int test_algorithms(void) {
    {
        const double close[6] = {1.0, 10.0, 1.1, 11.0, 1.2, 12.0};
        const double volume[6] = {10.0, 100.0, 20.0, 200.0, 30.0, 300.0};
        double out[6];
        CHECK(rflux_factor_momentum(close, 3, 2, 1, out) == RFLUX_OK);
        CHECK(isnan(out[0]) && isnan(out[1]));
        CHECK(approx(out[2], 0.1, 1e-12));
        CHECK(approx(out[3], 0.1, 1e-12));
        CHECK(rflux_factor_volume_reversal(close, volume, 3, 2, 1, out) ==
              RFLUX_OK);
        CHECK(approx(out[2], -0.1, 1e-12));
        CHECK(rflux_factor_vwap_deviation(close, volume, 3, 2, 2, out) ==
              RFLUX_OK);
        CHECK(rflux_factor_amihud(close, volume, 3, 2, 1, 1e6, out) ==
              RFLUX_OK);
        CHECK(rflux_factor_momentum(NULL, 3, 2, 1, out) ==
              RFLUX_ERR_INVALID_ARGUMENT);
    }
    {
        const double covariance[4] = {0.04, 0.006, 0.006, 0.09};
        double weights[2];
        size_t order[2];
        CHECK(rflux_hrp_weights(covariance, 2, weights) == RFLUX_OK);
        CHECK(approx(weights[0] + weights[1], 1.0, 1e-12));
        CHECK(approx(weights[0], 0.6923076923076923, 1e-12));
        CHECK(rflux_hrp_order(covariance, 2, order) == RFLUX_OK);
        CHECK((order[0] == 0 && order[1] == 1) ||
              (order[0] == 1 && order[1] == 0));
    }
    {
        double trades[10], cost, variance, kappa;
        CHECK(rflux_almgren_chriss_plan(100.0, 10, 1.0, 0.01, 0.001, 0.02,
                                        0.0, trades, &cost, &variance,
                                        &kappa) == RFLUX_OK);
        CHECK(approx(cost, 105.0, 1e-9));
        double total = 0.0;
        for (int i = 0; i < 10; ++i) total += trades[i];
        CHECK(approx(total, 100.0, 1e-9));
    }
    {
        const double features[5] = {0, 1, 2, 3, 4};
        const double target[5] = {1, 3, 5, 7, 9};
        double coefficients[1], intercept, scale;
        size_t iter;
        int32_t converged;
        CHECK(rflux_huber_regression(features, target, 5, 1, 1.35, 100,
                                     1e-8, 1, coefficients, &intercept,
                                     &scale, &iter, &converged) == RFLUX_OK);
        CHECK(approx(coefficients[0], 2.0, 1e-9));
        CHECK(approx(intercept, 1.0, 1e-9));
    }
    {
        const double adjacency[4] = {0.0, 1.0, 1.0, 0.0};
        double giant, stddev, susceptibility, threshold;
        size_t largest;
        int32_t systemic;
        CHECK(rflux_site_percolation_analyze(
                  adjacency, 2, 0.5, 2048, 42, 0.5, 11, &giant, &stddev,
                  &susceptibility, &largest, &systemic) == RFLUX_OK);
        CHECK(approx(giant, 0.5, 0.03));
        CHECK(largest == 1 || largest == 2);
        CHECK(rflux_site_percolation_threshold(adjacency, 2, 512, 42, 0.5,
                                               11, &threshold) == RFLUX_OK);
        CHECK(threshold > 0.0 && threshold <= 1.0);
    }
    {
        int32_t allowed;
        char reason[128];
        CHECK(rflux_pretrade_check(10, 100.0, 0, 0.2, 0.05, 0.05, 0.5, 0.2,
                                   0.1, 0, &allowed, reason,
                                   sizeof(reason)) == RFLUX_OK);
        CHECK(allowed == 1);
        CHECK(rflux_pretrade_check(10, 100.0, 0, 0.8, 0.05, 0.05, 0.5, 0.2,
                                   0.1, 0, &allowed, reason,
                                   sizeof(reason)) == RFLUX_OK);
        CHECK(allowed == 0);
        CHECK(strlen(reason) > 0);
    }
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_status();
    failures += test_indicators();
    failures += test_stats();
    failures += test_ranking();
    failures += test_hmm();
    failures += test_regimes();
    failures += test_graphs();
    failures += test_algorithms();
    if (failures == 0) {
        printf("test_capi: all C ABI tests passed\n");
        return 0;
    }
    fprintf(stderr, "test_capi: %d test group(s) failed\n", failures);
    return 1;
}
