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

static int test_status(void) {
    double out[5];
    const double x[5] = {1, 2, 3, 4, 5};
    CHECK(rflux_abi_version() == RFLUX_ABI_VERSION);
    CHECK(strcmp(rflux_version_string(), "0.4.2") == 0);
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
    CHECK(n_components == 2 && largest == 2);
    CHECK(labels[0] == labels[1] && labels[2] == labels[3]);
    CHECK(labels[0] != labels[2]);

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
        const double complete[16] = {
            0, 1, 1, 1,
            1, 0, 1, 1,
            1, 1, 0, 1,
            1, 1, 1, 0};
        const double probabilities[2] = {0.0, 1.0};
        double giant[2], giant_std[2], susceptibility[2];
        CHECK(rflux_bond_percolation(complete, 4, probabilities, 2, 20, 3,
                                     giant, giant_std, susceptibility) ==
              RFLUX_OK);
        CHECK(approx(giant[0], 0.25, 1e-12));
        CHECK(approx(giant[1], 1.0, 1e-12));
        CHECK(approx(giant_std[0], 0.0, 1e-6));
        CHECK(approx(susceptibility[0], 0.75, 1e-12));
        CHECK(approx(susceptibility[1], 0.0, 1e-12));

        CHECK(rflux_site_percolation(complete, 4, probabilities, 2, 20, 3,
                                     giant, giant_std, susceptibility) ==
              RFLUX_OK);
        CHECK(approx(giant[0], 0.0, 1e-12));
        CHECK(approx(giant[1], 1.0, 1e-12));
    }

    {
        const double complete[16] = {
            0, 1, 1, 1,
            1, 0, 1, 1,
            1, 1, 0, 1,
            1, 1, 1, 0};
        const uint32_t seeds[1] = {0};
        double curve[4];
        double frequency[4];
        CHECK(rflux_risk_diffusion(complete, 4, seeds, 1, 10.0, 0.0, 3, 5, 9,
                                   curve, frequency) == RFLUX_OK);
        CHECK(approx(curve[0], 0.25, 1e-12));
        CHECK(curve[1] > 0.9);
        CHECK(approx(frequency[0], 1.0, 1e-12));
        for (size_t i = 0; i < 4; ++i) {
            CHECK(frequency[i] >= 0.0 && frequency[i] <= 1.0);
        }
        CHECK(rflux_risk_diffusion(complete, 4, seeds, 1, 10.0, 0.0,
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

int main(void) {
    int failures = 0;
    failures += test_status();
    failures += test_indicators();
    failures += test_stats();
    failures += test_ranking();
    failures += test_hmm();
    failures += test_regimes();
    failures += test_graphs();
    if (failures == 0) {
        printf("test_capi: all C ABI tests passed\n");
        return 0;
    }
    fprintf(stderr, "test_capi: %d test group(s) failed\n", failures);
    return 1;
}
