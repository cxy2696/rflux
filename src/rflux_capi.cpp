#include "rflux/rflux.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <new>
#include <string>
#include <vector>

#include "internal/common.hpp"
#include "rflux/percolab/network.hpp"
#include "rflux/percolab/percolation.hpp"
#include "rflux/qflux/indicators.hpp"
#include "rflux/qflux/ranking.hpp"
#include "rflux/qflux/stats.hpp"
#include "rflux/qflux/window.hpp"
#include "rflux/regimelab/hmm.hpp"
#include "rflux/regimelab/regime.hpp"

struct rflux_hmm {
    rflux_hmm(std::size_t n_states, std::size_t n_features)
        : model(n_states, n_features) {}
    rflux::regimelab::GaussianHMM model;
};

namespace {

thread_local char g_last_error[1024] = {0};

const char* status_name(int status) noexcept {
    switch (status) {
        case RFLUX_OK:
            return "RFLUX_OK";
        case RFLUX_ERR_INVALID_ARGUMENT:
            return "RFLUX_ERR_INVALID_ARGUMENT";
        case RFLUX_ERR_OUT_OF_MEMORY:
            return "RFLUX_ERR_OUT_OF_MEMORY";
        case RFLUX_ERR_NONFINITE:
            return "RFLUX_ERR_NONFINITE";
        case RFLUX_ERR_MODEL_NOT_READY:
            return "RFLUX_ERR_MODEL_NOT_READY";
        case RFLUX_ERR_NUMERICAL:
            return "RFLUX_ERR_NUMERICAL";
        case RFLUX_ERR_INTERNAL:
            return "RFLUX_ERR_INTERNAL";
        default:
            return "RFLUX_ERR_UNKNOWN";
    }
}

void record_error(const char* function, const char* message) noexcept {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s: %s",
                  function != nullptr ? function : "rflux",
                  message != nullptr ? message : "");
}

void clear_error() noexcept { g_last_error[0] = '\0'; }

template <typename Function>
rflux_status api_guard(const char* function, Function&& function_call) noexcept {
    clear_error();
    try {
        function_call();
        return RFLUX_OK;
    } catch (const rflux::NonFinite& error) {
        record_error(function, error.what());
        return RFLUX_ERR_NONFINITE;
    } catch (const rflux::InvalidArgument& error) {
        record_error(function, error.what());
        return RFLUX_ERR_INVALID_ARGUMENT;
    } catch (const rflux::NotReady& error) {
        record_error(function, error.what());
        return RFLUX_ERR_MODEL_NOT_READY;
    } catch (const rflux::NumericalError& error) {
        record_error(function, error.what());
        return RFLUX_ERR_NUMERICAL;
    } catch (const std::bad_alloc&) {
        record_error(function, "allocation failed");
        return RFLUX_ERR_OUT_OF_MEMORY;
    } catch (const std::exception& error) {
        record_error(function, error.what());
        return RFLUX_ERR_INTERNAL;
    } catch (...) {
        record_error(function, "unknown exception");
        return RFLUX_ERR_INTERNAL;
    }
}

void require_output(const void* pointer, const char* name) {
    if (pointer == nullptr) {
        throw rflux::InvalidArgument(std::string(name) +
                                     " output must not be NULL");
    }
}

template <typename Container>
void copy_out(const Container& values, double* out) {
    std::copy(values.begin(), values.end(), out);
}

template <typename Container>
void copy_out_i32(const Container& values, std::int32_t* out) {
    std::copy(values.begin(), values.end(), out);
}

template <typename Function>
rflux_status guarded_vector(const char* name, double* out,
                            const char* out_name, Function&& function) {
    return api_guard(name, [&] {
        require_output(out, out_name);
        const auto values = function();
        copy_out(values, out);
    });
}

template <typename Function>
rflux_status guarded_i32(const char* name, std::int32_t* out,
                         const char* out_name, Function&& function) {
    return api_guard(name, [&] {
        require_output(out, out_name);
        const auto values = function();
        copy_out_i32(values, out);
    });
}

template <typename Function>
rflux_status guarded_scalar(const char* name, double* out,
                            const char* out_name, Function&& function) {
    return api_guard(name, [&] {
        require_output(out, out_name);
        *out = function();
    });
}

template <typename Function>
rflux_status guarded_percolation(const char* name, double* giant_mean,
                                 double* giant_std, double* susceptibility,
                                 Function&& function) {
    return api_guard(name, [&] {
        require_output(giant_mean, "giant_mean");
        require_output(giant_std, "giant_std");
        require_output(susceptibility, "susceptibility");
        const auto points = function();
        for (std::size_t i = 0; i < points.size(); ++i) {
            giant_mean[i] = points[i].giant_mean;
            giant_std[i] = points[i].giant_std;
            susceptibility[i] = points[i].susceptibility;
        }
    });
}

template <typename Prepare, typename Function>
rflux_status guarded_vector(const char* name, double* out,
                            const char* out_name, Prepare&& prepare,
                            Function&& function) {
    return api_guard(name, [&] {
        prepare();
        require_output(out, out_name);
        const auto values = function();
        copy_out(values, out);
    });
}

template <typename Prepare, typename Function>
rflux_status guarded_i32(const char* name, std::int32_t* out,
                         const char* out_name, Prepare&& prepare,
                         Function&& function) {
    return api_guard(name, [&] {
        prepare();
        require_output(out, out_name);
        const auto values = function();
        copy_out_i32(values, out);
    });
}

rflux::regimelab::HmmOptions parse_hmm_options(
    const rflux_hmm_options* options) {
    rflux::regimelab::HmmOptions parsed;
    if (options == nullptr) return parsed;
    if (options->struct_size < sizeof(rflux_hmm_options)) {
        throw rflux::InvalidArgument(
            "options->struct_size is smaller than the ABI-v1 options struct");
    }
    if (options->reserved0 != 0) {
        throw rflux::InvalidArgument("options->reserved0 must be zero");
    }
    if (options->sort_states != 0 && options->sort_states != 1) {
        throw rflux::InvalidArgument("options->sort_states must be 0 or 1");
    }
    parsed.max_iter = static_cast<std::size_t>(options->max_iter);
    parsed.n_init = static_cast<std::size_t>(options->n_init);
    parsed.tol = options->tol;
    parsed.variance_floor = options->variance_floor;
    parsed.seed = options->seed;
    parsed.sort_states = options->sort_states != 0;
    return parsed;
}

rflux_hmm* require_model(rflux_hmm* model, const char* what) {
    if (model == nullptr) {
        throw rflux::InvalidArgument(std::string(what) +
                                     " must not be NULL");
    }
    return model;
}

const rflux_hmm* require_model(const rflux_hmm* model, const char* what) {
    if (model == nullptr) {
        throw rflux::InvalidArgument(std::string(what) +
                                     " must not be NULL");
    }
    return model;
}

}  // namespace

extern "C" {

uint32_t rflux_abi_version(void) { return RFLUX_ABI_VERSION; }

const char* rflux_version_string(void) {
#define RFLUX_STRINGIFY_INNER(value) #value
#define RFLUX_STRINGIFY(value) RFLUX_STRINGIFY_INNER(value)
    return RFLUX_STRINGIFY(RFLUX_VERSION_MAJOR) "." RFLUX_STRINGIFY(
               RFLUX_VERSION_MINOR) "." RFLUX_STRINGIFY(RFLUX_VERSION_PATCH);
#undef RFLUX_STRINGIFY
#undef RFLUX_STRINGIFY_INNER
}

const char* rflux_status_name(int status) {
    return status_name(status);
}

const char* rflux_last_error(void) { return g_last_error; }

void rflux_clear_error(void) { clear_error(); }

/* ---------------------------------------------------------------------- */
/* qflux indicators                                                       */
/* ---------------------------------------------------------------------- */

rflux_status rflux_returns(const double* x, size_t n, int32_t log_returns,
                           double* out) {
    return guarded_vector("rflux_returns", out, "out",
                          [&] { return rflux::qflux::returns(x, n, log_returns != 0); });
}

rflux_status rflux_sma(const double* x, size_t n, size_t window, double* out) {
    return guarded_vector("rflux_sma", out, "out",
                          [&] { return rflux::qflux::sma(x, n, window); });
}

rflux_status rflux_rolling_sum(const double* x, size_t n, size_t window,
                               double* out) {
    return guarded_vector("rflux_rolling_sum", out, "out",
                          [&] { return rflux::qflux::rolling_sum(x, n, window); });
}

rflux_status rflux_rolling_std(const double* x, size_t n, size_t window,
                               size_t ddof, double* out) {
    return guarded_vector("rflux_rolling_std", out, "out", [&] {
        return rflux::qflux::rolling_std(x, n, window, ddof);
    });
}

rflux_status rflux_rolling_min(const double* x, size_t n, size_t window,
                               double* out) {
    return guarded_vector("rflux_rolling_min", out, "out",
                          [&] { return rflux::qflux::rolling_min(x, n, window); });
}

rflux_status rflux_rolling_max(const double* x, size_t n, size_t window,
                               double* out) {
    return guarded_vector("rflux_rolling_max", out, "out",
                          [&] { return rflux::qflux::rolling_max(x, n, window); });
}

rflux_status rflux_rolling_zscore(const double* x, size_t n, size_t window,
                                  size_t ddof, double* out) {
    return guarded_vector("rflux_rolling_zscore", out, "out", [&] {
        return rflux::qflux::rolling_zscore(x, n, window, ddof);
    });
}

rflux_status rflux_ema(const double* x, size_t n, size_t span, double* out) {
    return guarded_vector("rflux_ema", out, "out",
                          [&] { return rflux::qflux::ema(x, n, span); });
}

rflux_status rflux_rsi(const double* x, size_t n, size_t period, double* out) {
    return guarded_vector("rflux_rsi", out, "out",
                          [&] { return rflux::qflux::rsi(x, n, period); });
}

rflux_status rflux_macd(const double* x, size_t n, size_t fast, size_t slow,
                        size_t signal, double* macd, double* signal_line,
                        double* histogram) {
    return api_guard("rflux_macd", [&] {
        require_output(macd, "macd");
        require_output(signal_line, "signal_line");
        require_output(histogram, "histogram");
        const rflux::qflux::MacdResult result =
            rflux::qflux::macd(x, n, fast, slow, signal);
        copy_out(result.macd, macd);
        copy_out(result.signal, signal_line);
        copy_out(result.histogram, histogram);
    });
}

rflux_status rflux_bollinger(const double* x, size_t n, size_t window,
                             double n_std, double* middle, double* upper,
                             double* lower) {
    return api_guard("rflux_bollinger", [&] {
        require_output(middle, "middle");
        require_output(upper, "upper");
        require_output(lower, "lower");
        const rflux::qflux::BollingerBands bands =
            rflux::qflux::bollinger(x, n, window, n_std);
        copy_out(bands.middle, middle);
        copy_out(bands.upper, upper);
        copy_out(bands.lower, lower);
    });
}

rflux_status rflux_atr(const double* high, const double* low,
                       const double* close, size_t n, size_t period,
                       double* out) {
    return guarded_vector("rflux_atr", out, "out",
                          [&] { return rflux::qflux::atr(high, low, close, n, period); });
}

rflux_status rflux_drawdown(const double* equity, size_t n, double* out) {
    return guarded_vector("rflux_drawdown", out, "out",
                          [&] { return rflux::qflux::drawdown(equity, n); });
}

rflux_status rflux_rolling_corr(const double* x, const double* y, size_t n,
                                size_t window, double* out) {
    return guarded_vector("rflux_rolling_corr", out, "out",
                          [&] { return rflux::qflux::rolling_corr(x, y, n, window); });
}

rflux_status rflux_rolling_beta(const double* y, const double* x, size_t n,
                                size_t window, double* out) {
    return guarded_vector("rflux_rolling_beta", out, "out",
                          [&] { return rflux::qflux::rolling_beta(y, x, n, window); });
}

rflux_status rflux_rolling_slope(const double* x, size_t n, size_t window,
                                 double* slope) {
    return guarded_vector("rflux_rolling_slope", slope, "slope",
                          [&] { return rflux::qflux::rolling_slope(x, n, window); });
}

rflux_status rflux_rolling_r2(const double* x, size_t n, size_t window,
                              double* r2) {
    return guarded_vector("rflux_rolling_r2", r2, "r2",
                          [&] { return rflux::qflux::rolling_r2(x, n, window); });
}

/* ---------------------------------------------------------------------- */
/* qflux statistics                                                       */
/* ---------------------------------------------------------------------- */

rflux_status rflux_describe(const double* x, size_t n, double* mean,
                            double* variance, double* stddev, double* skewness,
                            double* excess_kurtosis, size_t* n_valid) {
    return api_guard("rflux_describe", [&] {
        require_output(mean, "mean");
        require_output(variance, "variance");
        require_output(stddev, "stddev");
        require_output(skewness, "skewness");
        require_output(excess_kurtosis, "excess_kurtosis");
        require_output(n_valid, "n_valid");
        const rflux::qflux::Moments moments = rflux::qflux::describe(x, n);
        *mean = moments.mean;
        *variance = moments.variance;
        *stddev = moments.stddev;
        *skewness = moments.skewness;
        *excess_kurtosis = moments.excess_kurtosis;
        *n_valid = moments.n_valid;
    });
}

rflux_status rflux_quantile(const double* x, size_t n, double q, double* out) {
    return guarded_scalar("rflux_quantile", out, "out",
                          [&] { return rflux::qflux::quantile(x, n, q); });
}

rflux_status rflux_pearson(const double* x, const double* y, size_t n,
                           double* out) {
    return guarded_scalar("rflux_pearson", out, "out",
                          [&] { return rflux::qflux::pearson(x, y, n); });
}

rflux_status rflux_corr_matrix(const double* panel, size_t n_obs,
                               size_t n_assets, double* corr_matrix) {
    return guarded_vector("rflux_corr_matrix", corr_matrix, "corr_matrix",
                          [&] { return rflux::qflux::correlation_matrix(panel, n_obs, n_assets); });
}

rflux_status rflux_sharpe(const double* returns, size_t n,
                          double periods_per_year, double risk_free,
                          double* out) {
    return guarded_scalar("rflux_sharpe", out, "out", [&] {
        return rflux::qflux::sharpe(returns, n, periods_per_year, risk_free);
    });
}

rflux_status rflux_sortino(const double* returns, size_t n,
                           double periods_per_year, double risk_free,
                           double* out) {
    return guarded_scalar("rflux_sortino", out, "out", [&] {
        return rflux::qflux::sortino(returns, n, periods_per_year, risk_free);
    });
}

rflux_status rflux_var_cvar(const double* returns, size_t n, double confidence,
                            double* var, double* cvar) {
    return api_guard("rflux_var_cvar", [&] {
        require_output(var, "var");
        require_output(cvar, "cvar");
        const rflux::qflux::TailRisk risk =
            rflux::qflux::historical_var_cvar(returns, n, confidence);
        *var = risk.var;
        *cvar = risk.cvar;
    });
}

/* ---------------------------------------------------------------------- */
/* qflux ranking                                                          */
/* ---------------------------------------------------------------------- */

rflux_status rflux_rank(const double* values, size_t n, double* out) {
    return guarded_vector("rflux_rank", out, "out",
                          [&] { return rflux::qflux::rank(values, n); });
}

rflux_status rflux_percentile_rank(const double* values, size_t n,
                                   double* out) {
    return guarded_vector("rflux_percentile_rank", out, "out",
                          [&] { return rflux::qflux::percentile_rank(values, n); });
}

rflux_status rflux_zscore(const double* values, size_t n, size_t ddof,
                          double* out) {
    return guarded_vector("rflux_zscore", out, "out",
                          [&] { return rflux::qflux::zscore(values, n, ddof); });
}

rflux_status rflux_winsorize(const double* values, size_t n, double lower_q,
                             double upper_q, double* out) {
    return guarded_vector("rflux_winsorize", out, "out",
                          [&] { return rflux::qflux::winsorize(values, n, lower_q, upper_q); });
}

rflux_status rflux_neutralize(const double* values, size_t n,
                              const int32_t* groups, size_t group_count,
                              double* out) {
    return guarded_vector("rflux_neutralize", out, "out", [&] {
        return rflux::qflux::neutralize(values, n, groups, group_count);
    });
}

/* ---------------------------------------------------------------------- */
/* regimelab HMM                                                          */
/* ---------------------------------------------------------------------- */

rflux_status rflux_hmm_options_init(rflux_hmm_options* options,
                                    size_t options_size) {
    return api_guard("rflux_hmm_options_init", [&] {
        require_output(options, "options");
        if (options_size < sizeof(rflux_hmm_options)) {
            throw rflux::InvalidArgument(
                "options_size is smaller than rflux_hmm_options");
        }
        std::memset(options, 0, sizeof(rflux_hmm_options));
        options->struct_size = sizeof(rflux_hmm_options);
        options->max_iter = 200;
        options->n_init = 3;
        options->tol = 1e-6;
        options->variance_floor = 1e-6;
        options->seed = 42;
        options->sort_states = 1;
    });
}

rflux_status rflux_hmm_create(size_t n_states, size_t n_features,
                              rflux_hmm** out) {
    return api_guard("rflux_hmm_create", [&] {
        require_output(out, "out");
        *out = nullptr;
        *out = new rflux_hmm(n_states, n_features);
    });
}

void rflux_hmm_destroy(rflux_hmm* model) { delete model; }

rflux_status rflux_hmm_dimensions(const rflux_hmm* model, size_t* n_states,
                                  size_t* n_features) {
    return api_guard("rflux_hmm_dimensions", [&] {
        require_model(model, "model");
        require_output(n_states, "n_states");
        require_output(n_features, "n_features");
        *n_states = model->model.n_states();
        *n_features = model->model.n_features();
    });
}

rflux_status rflux_hmm_fit(rflux_hmm* model, const double* observations,
                           size_t n_obs, const rflux_hmm_options* options) {
    return api_guard("rflux_hmm_fit", [&] {
        require_model(model, "model");
        model->model.fit(observations, n_obs, parse_hmm_options(options));
    });
}

rflux_status rflux_hmm_fit_summary(const rflux_hmm* model, size_t* iterations,
                                   int32_t* converged, double* log_likelihood,
                                   double* aic, double* bic) {
    return api_guard("rflux_hmm_fit_summary", [&] {
        require_model(model, "model");
        require_output(iterations, "iterations");
        require_output(converged, "converged");
        require_output(log_likelihood, "log_likelihood");
        require_output(aic, "aic");
        require_output(bic, "bic");
        *iterations = model->model.iterations();
        *converged = model->model.converged() ? 1 : 0;
        *log_likelihood = model->model.log_likelihood();
        *aic = model->model.aic();
        *bic = model->model.bic();
    });
}

rflux_status rflux_hmm_score(const rflux_hmm* model,
                              const double* observations, size_t n_obs,
                              double* log_likelihood) {
    return api_guard("rflux_hmm_score", [&] {
        require_model(model, "model");
        require_output(log_likelihood, "log_likelihood");
        *log_likelihood = model->model.score(observations, n_obs);
    });
}

rflux_status rflux_hmm_get_parameters(const rflux_hmm* model, double* start_prob,
                                      double* transition, double* means,
                                      double* variances) {
    return api_guard("rflux_hmm_get_parameters", [&] {
        require_model(model, "model");
        require_output(start_prob, "start_prob");
        require_output(transition, "transition");
        require_output(means, "means");
        require_output(variances, "variances");
        const rflux::regimelab::HmmParameters& parameters =
            model->model.parameters();
        copy_out(parameters.start_prob, start_prob);
        copy_out(parameters.transition, transition);
        copy_out(parameters.means, means);
        copy_out(parameters.variances, variances);
    });
}

rflux_status rflux_hmm_set_parameters(rflux_hmm* model,
                                      const double* start_prob,
                                      const double* transition,
                                      const double* means,
                                      const double* variances) {
    return api_guard("rflux_hmm_set_parameters", [&] {
        require_model(model, "model");
        if (start_prob == nullptr || transition == nullptr || means == nullptr ||
            variances == nullptr) {
            throw rflux::InvalidArgument("HMM parameter buffers must not be NULL");
        }
        rflux::regimelab::HmmParameters parameters;
        const std::size_t k = model->model.n_states();
        const std::size_t d = model->model.n_features();
        parameters.start_prob.assign(start_prob, start_prob + k);
        parameters.transition.assign(transition, transition + k * k);
        parameters.means.assign(means, means + k * d);
        parameters.variances.assign(variances, variances + k * d);
        model->model.set_parameters(parameters);
    });
}

rflux_status rflux_hmm_viterbi(const rflux_hmm* model,
                               const double* observations, size_t n_obs,
                               int32_t* states) {
    return guarded_i32(
        "rflux_hmm_viterbi", states, "states",
        [&] { require_model(model, "model"); },
        [&] { return model->model.viterbi(observations, n_obs); });
}

rflux_status rflux_hmm_filtered(const rflux_hmm* model,
                                const double* observations, size_t n_obs,
                                double* probs) {
    return guarded_vector(
        "rflux_hmm_filtered", probs, "probs",
        [&] { require_model(model, "model"); },
        [&] { return model->model.filtered(observations, n_obs); });
}

rflux_status rflux_hmm_smoothed(const rflux_hmm* model,
                                const double* observations, size_t n_obs,
                                double* probs) {
    return guarded_vector(
        "rflux_hmm_smoothed", probs, "probs",
        [&] { require_model(model, "model"); },
        [&] { return model->model.smoothed(observations, n_obs); });
}

rflux_status rflux_hmm_forecast(const rflux_hmm* model,
                                const double* state_probs, size_t horizon,
                                double* out) {
    return guarded_vector(
        "rflux_hmm_forecast", out, "out",
        [&] { require_model(model, "model"); },
        [&] {
            return model->model.forecast(state_probs, model->model.n_states(),
                                         horizon);
        });
}

/* ---------------------------------------------------------------------- */
/* regimelab regimes                                                      */
/* ---------------------------------------------------------------------- */

rflux_status rflux_volatility_regime(const double* returns, size_t n,
                                     size_t vol_window, size_t lookback,
                                     double low_q, double high_q,
                                     int32_t* labels, double* volatility) {
    return api_guard("rflux_volatility_regime", [&] {
        require_output(labels, "labels");
        require_output(volatility, "volatility");
        const rflux::regimelab::VolatilityRegimeResult result =
            rflux::regimelab::volatility_regime(returns, n, vol_window,
                                                lookback, low_q, high_q);
        copy_out(result.volatility, volatility);
        copy_out_i32(result.labels, labels);
    });
}

rflux_status rflux_trend_regime(const double* prices, size_t n,
                                size_t fast_window, size_t slow_window,
                                double threshold, int32_t* labels,
                                double* fast_ma, double* slow_ma) {
    return api_guard("rflux_trend_regime", [&] {
        require_output(labels, "labels");
        require_output(fast_ma, "fast_ma");
        require_output(slow_ma, "slow_ma");
        const rflux::regimelab::TrendRegimeResult result =
            rflux::regimelab::trend_regime(prices, n, fast_window, slow_window,
                                           threshold);
        copy_out(result.fast_ma, fast_ma);
        copy_out(result.slow_ma, slow_ma);
        copy_out_i32(result.labels, labels);
    });
}

rflux_status rflux_composite_regime(const int32_t* vol_labels,
                                    const int32_t* trend_labels, size_t n,
                                    int32_t* out) {
    return guarded_i32("rflux_composite_regime", out, "out", [&] {
        return rflux::regimelab::composite_regime(vol_labels, trend_labels, n);
    });
}

/* ---------------------------------------------------------------------- */
/* percolab graphs                                                        */
/* ---------------------------------------------------------------------- */

rflux_status rflux_corr_to_distance(const double* corr, size_t n, double* out) {
    return guarded_vector("rflux_corr_to_distance", out, "out",
                          [&] { return rflux::percolab::correlation_to_distance(corr, n); });
}

rflux_status rflux_graph_components(const double* weights, size_t n,
                                    double threshold, int32_t* labels,
                                    size_t* n_components,
                                    size_t* largest_size) {
    return api_guard("rflux_graph_components", [&] {
        require_output(labels, "labels");
        require_output(n_components, "n_components");
        require_output(largest_size, "largest_size");
        const rflux::percolab::ComponentSummary summary =
            rflux::percolab::connected_components(weights, n, threshold);
        copy_out_i32(summary.labels, labels);
        *n_components = summary.n_components;
        *largest_size = summary.largest_size;
    });
}

rflux_status rflux_network_strength(const double* weights, size_t n,
                                    double* out) {
    return guarded_vector("rflux_network_strength", out, "out",
                          [&] { return rflux::percolab::strength(weights, n); });
}

rflux_status rflux_network_centrality(const double* weights, size_t n,
                                      uint32_t kind, double* out) {
    return guarded_vector("rflux_network_centrality", out, "out", [&] {
        if (kind == RFLUX_CENTRALITY_DEGREE) {
            return rflux::percolab::strength(weights, n);
        }
        if (kind == RFLUX_CENTRALITY_EIGENVECTOR) {
            return rflux::percolab::eigenvector_centrality(weights, n);
        }
        throw rflux::InvalidArgument(
            "centrality kind must be RFLUX_CENTRALITY_DEGREE or "
            "RFLUX_CENTRALITY_EIGENVECTOR");
    });
}

rflux_status rflux_network_clustering(const double* weights, size_t n,
                                      double* out) {
    return guarded_vector("rflux_network_clustering", out, "out", [&] {
        return rflux::percolab::clustering_coefficients(weights, n);
    });
}

rflux_status rflux_network_mst(const double* distance, size_t n,
                               uint32_t* edge_u, uint32_t* edge_v,
                               double* edge_weight, size_t* n_edges) {
    return api_guard("rflux_network_mst", [&] {
        require_output(n_edges, "n_edges");
        const std::vector<rflux::percolab::MstEdge> tree =
            rflux::percolab::minimum_spanning_tree(distance, n);
        if (!tree.empty()) {
            require_output(edge_u, "edge_u");
            require_output(edge_v, "edge_v");
            require_output(edge_weight, "edge_weight");
        }
        for (std::size_t i = 0; i < tree.size(); ++i) {
            edge_u[i] = tree[i].u;
            edge_v[i] = tree[i].v;
            edge_weight[i] = tree[i].weight;
        }
        *n_edges = tree.size();
    });
}

rflux_status rflux_bond_percolation(
    const double* adjacency, size_t n, const double* probabilities,
    size_t n_probabilities, size_t repetitions, uint64_t seed,
    double* giant_mean, double* giant_std, double* susceptibility) {
    return guarded_percolation(
        "rflux_bond_percolation", giant_mean, giant_std, susceptibility, [&] {
            return rflux::percolab::bond_percolation(
                adjacency, n, probabilities, n_probabilities, repetitions,
                seed);
        });
}

rflux_status rflux_site_percolation(
    const double* adjacency, size_t n, const double* probabilities,
    size_t n_probabilities, size_t repetitions, uint64_t seed,
    double* giant_mean, double* giant_std, double* susceptibility) {
    return guarded_percolation(
        "rflux_site_percolation", giant_mean, giant_std, susceptibility, [&] {
            return rflux::percolab::site_percolation(
                adjacency, n, probabilities, n_probabilities, repetitions,
                seed);
        });
}

rflux_status rflux_risk_diffusion(
    const double* adjacency, size_t n, const uint32_t* seed_nodes,
    size_t n_seed_nodes, double beta, double gamma, size_t steps,
    size_t repetitions, uint64_t seed, double* mean_affected,
    double* infection_frequency) {
    return api_guard("rflux_risk_diffusion", [&] {
        require_output(mean_affected, "mean_affected");
        require_output(infection_frequency, "infection_frequency");
        const rflux::percolab::DiffusionResult result =
            rflux::percolab::risk_diffusion(
                adjacency, n, seed_nodes, n_seed_nodes, beta, gamma, steps,
                repetitions, seed);
        copy_out(result.mean_affected, mean_affected);
        copy_out(result.infection_frequency, infection_frequency);
    });
}

}  // extern "C"
