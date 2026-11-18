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
#include "rflux/percolab/detector.hpp"
#include "rflux/percolab/percolation.hpp"
#include "rflux/qflux/indicators.hpp"
#include "rflux/qflux/ranking.hpp"
#include "rflux/qflux/stats.hpp"
#include "rflux/qflux/window.hpp"
#include "rflux/regimelab/hmm.hpp"
#include "rflux/regimelab/regime.hpp"
#include "rflux/execution/almgren_chriss.hpp"
#include "rflux/factorlab/evaluation.hpp"
#include "rflux/factorlab/factors.hpp"
#include "rflux/mllab/huber.hpp"
#include "rflux/portfoliolab/hrp.hpp"
#include "rflux/risk/pretrade.hpp"
#include "rflux/risk/tail_dependence.hpp"

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
    } catch (const std::invalid_argument& error) {
        record_error(function, error.what());
        return RFLUX_ERR_INVALID_ARGUMENT;
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

/* Buffer-first variant: the kernel writes directly into the caller's array, so
 * the C ABI never materialises an intermediate std::vector. */
template <typename Function>
rflux_status guarded_into(const char* name, double* out, const char* out_name,
                          Function&& function) {
    return api_guard(name, [&] {
        require_output(out, out_name);
        function(out);
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

/* HMM inference entry points share one handle check and one copy-out path. */
template <typename Function>
rflux_status guarded_model_vector(const rflux_hmm* model, const char* name,
                                  double* out, const char* out_name,
                                  Function&& function) {
    return api_guard(name, [&] {
        require_model(model, "model");
        require_output(out, out_name);
        const auto values = function();
        copy_out(values, out);
    });
}

template <typename Function>
rflux_status guarded_model_i32(const rflux_hmm* model, const char* name,
                               std::int32_t* out, const char* out_name,
                               Function&& function) {
    return api_guard(name, [&] {
        require_model(model, "model");
        require_output(out, out_name);
        const auto values = function();
        copy_out_i32(values, out);
    });
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
    return guarded_into("rflux_returns", out, "out", [&](double* o) {
        rflux::qflux::returns_into(x, n, log_returns != 0, o);
    });
}

rflux_status rflux_sma(const double* x, size_t n, size_t window, double* out) {
    return guarded_into("rflux_sma", out, "out", [&](double* o) {
        rflux::qflux::sma_into(x, n, window, o);
    });
}

rflux_status rflux_rolling_sum(const double* x, size_t n, size_t window,
                               double* out) {
    return guarded_into("rflux_rolling_sum", out, "out", [&](double* o) {
        rflux::qflux::rolling_sum_into(x, n, window, o);
    });
}

rflux_status rflux_rolling_std(const double* x, size_t n, size_t window,
                               size_t ddof, double* out) {
    return guarded_into("rflux_rolling_std", out, "out", [&](double* o) {
        rflux::qflux::rolling_std_into(x, n, window, ddof, o);
    });
}

rflux_status rflux_rolling_min(const double* x, size_t n, size_t window,
                               double* out) {
    return guarded_into("rflux_rolling_min", out, "out", [&](double* o) {
        rflux::qflux::rolling_min_into(x, n, window, o);
    });
}

rflux_status rflux_rolling_max(const double* x, size_t n, size_t window,
                               double* out) {
    return guarded_into("rflux_rolling_max", out, "out", [&](double* o) {
        rflux::qflux::rolling_max_into(x, n, window, o);
    });
}

rflux_status rflux_rolling_zscore(const double* x, size_t n, size_t window,
                                  size_t ddof, double* out) {
    return guarded_into("rflux_rolling_zscore", out, "out", [&](double* o) {
        rflux::qflux::rolling_zscore_into(x, n, window, ddof, o);
    });
}

rflux_status rflux_rolling_quantile(const double* x, size_t n, size_t window,
                                    double q, double* out) {
    return guarded_into("rflux_rolling_quantile", out, "out", [&](double* o) {
        rflux::qflux::rolling_quantile_into(x, n, window, q, o);
    });
}

rflux_status rflux_ema(const double* x, size_t n, size_t span, double* out) {
    return guarded_into("rflux_ema", out, "out", [&](double* o) {
        rflux::qflux::ema_into(x, n, span, o);
    });
}

rflux_status rflux_rsi(const double* x, size_t n, size_t period, double* out) {
    return guarded_into("rflux_rsi", out, "out", [&](double* o) {
        rflux::qflux::rsi_into(x, n, period, o);
    });
}

rflux_status rflux_macd(const double* x, size_t n, size_t fast, size_t slow,
                        size_t signal, double* macd, double* signal_line,
                        double* histogram) {
    return api_guard("rflux_macd", [&] {
        require_output(macd, "macd");
        require_output(signal_line, "signal_line");
        require_output(histogram, "histogram");
        rflux::qflux::macd_into(x, n, fast, slow, signal, macd, signal_line,
                                histogram);
    });
}

rflux_status rflux_bollinger(const double* x, size_t n, size_t window,
                             double n_std, double* middle, double* upper,
                             double* lower) {
    return api_guard("rflux_bollinger", [&] {
        require_output(middle, "middle");
        require_output(upper, "upper");
        require_output(lower, "lower");
        rflux::qflux::bollinger_into(x, n, window, n_std, middle, upper, lower);
    });
}

rflux_status rflux_atr(const double* high, const double* low,
                       const double* close, size_t n, size_t period,
                       double* out) {
    return guarded_into("rflux_atr", out, "out", [&](double* o) {
        rflux::qflux::atr_into(high, low, close, n, period, o);
    });
}

rflux_status rflux_drawdown(const double* equity, size_t n, double* out) {
    return guarded_into("rflux_drawdown", out, "out", [&](double* o) {
        rflux::qflux::drawdown_into(equity, n, o);
    });
}

rflux_status rflux_rolling_corr(const double* x, const double* y, size_t n,
                                size_t window, double* out) {
    return guarded_into("rflux_rolling_corr", out, "out", [&](double* o) {
        rflux::qflux::rolling_corr_into(x, y, n, window, o);
    });
}

rflux_status rflux_rolling_cov(const double* x, const double* y, size_t n,
                               size_t window, size_t ddof, double* out) {
    return guarded_into("rflux_rolling_cov", out, "out", [&](double* o) {
        rflux::qflux::rolling_cov_into(x, y, n, window, ddof, o);
    });
}

rflux_status rflux_rolling_beta(const double* y, const double* x, size_t n,
                                size_t window, double* out) {
    return guarded_into("rflux_rolling_beta", out, "out", [&](double* o) {
        rflux::qflux::rolling_beta_into(y, x, n, window, o);
    });
}

rflux_status rflux_rolling_slope(const double* x, size_t n, size_t window,
                                 double* slope) {
    return guarded_into("rflux_rolling_slope", slope, "slope",
                        [&](double* o) {
                            rflux::qflux::rolling_slope_into(x, n, window, o);
                        });
}

rflux_status rflux_rolling_r2(const double* x, size_t n, size_t window,
                              double* r2) {
    return guarded_into("rflux_rolling_r2", r2, "r2", [&](double* o) {
        rflux::qflux::rolling_r2_into(x, n, window, o);
    });
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
    return guarded_into("rflux_corr_matrix", corr_matrix, "corr_matrix",
                        [&](double* o) {
                            rflux::qflux::correlation_matrix_into(
                                panel, n_obs, n_assets, o);
                        });
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
    return guarded_into("rflux_rank", out, "out", [&](double* o) {
        rflux::qflux::rank_into(values, n, o);
    });
}

rflux_status rflux_percentile_rank(const double* values, size_t n,
                                   double* out) {
    return guarded_into("rflux_percentile_rank", out, "out", [&](double* o) {
        rflux::qflux::percentile_rank_into(values, n, o);
    });
}

rflux_status rflux_zscore(const double* values, size_t n, size_t ddof,
                          double* out) {
    return guarded_into("rflux_zscore", out, "out", [&](double* o) {
        rflux::qflux::zscore_into(values, n, ddof, o);
    });
}

rflux_status rflux_winsorize(const double* values, size_t n, double lower_q,
                             double upper_q, double* out) {
    return guarded_into("rflux_winsorize", out, "out", [&](double* o) {
        rflux::qflux::winsorize_into(values, n, lower_q, upper_q, o);
    });
}

rflux_status rflux_neutralize(const double* values, size_t n,
                              const int32_t* groups, size_t group_count,
                              double* out) {
    return guarded_into("rflux_neutralize", out, "out", [&](double* o) {
        rflux::qflux::neutralize_into(values, n, groups, group_count, o);
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
    return guarded_model_i32(
        model, "rflux_hmm_viterbi", states, "states",
        [&] { return model->model.viterbi(observations, n_obs); });
}

rflux_status rflux_hmm_filtered(const rflux_hmm* model,
                                const double* observations, size_t n_obs,
                                double* probs) {
    return guarded_model_vector(
        model, "rflux_hmm_filtered", probs, "probs",
        [&] { return model->model.filtered(observations, n_obs); });
}

rflux_status rflux_hmm_smoothed(const rflux_hmm* model,
                                const double* observations, size_t n_obs,
                                double* probs) {
    return guarded_model_vector(
        model, "rflux_hmm_smoothed", probs, "probs",
        [&] { return model->model.smoothed(observations, n_obs); });
}

rflux_status rflux_hmm_forecast(const rflux_hmm* model,
                                const double* state_probs, size_t horizon,
                                double* out) {
    return guarded_model_vector(
        model, "rflux_hmm_forecast", out, "out",
        [&] {
            return model->model.forecast(state_probs, model->model.n_states(),
                                         horizon);
        });
}

rflux_status rflux_hmm_filter_step(const rflux_hmm* model,
                                   const double* state_probs,
                                   const double* observation, double* out) {
    return guarded_model_vector(
        model, "rflux_hmm_filter_step", out, "out",
        [&] { return model->model.filter_step(state_probs, observation); });
}

rflux_status rflux_hmm_expected_durations(const rflux_hmm* model, double* out) {
    return guarded_model_vector(
        model, "rflux_hmm_expected_durations", out, "out",
        [&] { return model->model.expected_durations(); });
}

rflux_status rflux_hmm_stationary(const rflux_hmm* model, double* out) {
    return guarded_model_vector(
        model, "rflux_hmm_stationary", out, "out",
        [&] { return model->model.stationary_distribution(); });
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

rflux_status rflux_network_transitivity(const double* weights, size_t n,
                                        double* out) {
    return guarded_scalar("rflux_network_transitivity", out, "out", [&] {
        return rflux::percolab::transitivity(weights, n);
    });
}

rflux_status rflux_network_path_stats(const double* weights, size_t n,
                                      double* efficiency,
                                      double* average_path_length,
                                      size_t* diameter,
                                      size_t* n_reachable_pairs,
                                      int32_t* connected) {
    return api_guard("rflux_network_path_stats", [&] {
        require_output(efficiency, "efficiency");
        require_output(average_path_length, "average_path_length");
        require_output(diameter, "diameter");
        require_output(n_reachable_pairs, "n_reachable_pairs");
        require_output(connected, "connected");
        const rflux::percolab::PathStats stats =
            rflux::percolab::path_statistics(weights, n);
        *efficiency = stats.efficiency;
        *average_path_length = stats.average_path_length;
        *diameter = stats.diameter;
        *n_reachable_pairs = stats.n_reachable_pairs;
        *connected = stats.connected ? 1 : 0;
    });
}

rflux_status rflux_network_core_numbers(const double* weights, size_t n,
                                        int32_t* out) {
    return guarded_i32("rflux_network_core_numbers", out, "out", [&] {
        return rflux::percolab::core_numbers(weights, n);
    });
}

rflux_status rflux_network_assortativity(const double* weights, size_t n,
                                         double* out) {
    return guarded_scalar("rflux_network_assortativity", out, "out", [&] {
        return rflux::percolab::degree_assortativity(weights, n);
    });
}

rflux_status rflux_network_modularity(const double* weights, size_t n,
                                      const int32_t* labels, size_t n_labels,
                                      double* out) {
    return guarded_scalar("rflux_network_modularity", out, "out", [&] {
        return rflux::percolab::modularity(weights, n, labels, n_labels);
    });
}

rflux_status rflux_network_betweenness(const double* weights, size_t n,
                                      int32_t normalized, double* out) {
    return guarded_vector("rflux_network_betweenness", out, "out", [&] {
        return rflux::percolab::betweenness_centrality(
            weights, n, normalized != 0);
    });
}

rflux_status rflux_network_bridges(const double* weights, size_t n,
                                     uint32_t* edge_u, uint32_t* edge_v,
                                     size_t* n_edges) {
    return api_guard("rflux_network_bridges", [&] {
        require_output(n_edges, "n_edges");
        const std::vector<rflux::percolab::BridgeEdge> edges =
            rflux::percolab::bridges(weights, n);
        if (!edges.empty()) {
            require_output(edge_u, "edge_u");
            require_output(edge_v, "edge_v");
        }
        for (std::size_t i = 0; i < edges.size(); ++i) {
            edge_u[i] = edges[i].u;
            edge_v[i] = edges[i].v;
        }
        *n_edges = edges.size();
    });
}

rflux_status rflux_network_articulation_points(const double* weights,
                                               size_t n, int32_t* out) {
    return guarded_i32(
        "rflux_network_articulation_points", out, "out",
        [&] { return rflux::percolab::articulation_points(weights, n); });
}

rflux_status rflux_network_truss_numbers(const double* weights, size_t n,
                                         int32_t* out) {
    return guarded_i32(
        "rflux_network_truss_numbers", out, "out",
        [&] { return rflux::percolab::truss_numbers(weights, n); });
}

rflux_status rflux_pagerank(const double* adjacency, size_t n, double damping,
                             size_t max_iter, double tol, double* out) {
    return guarded_vector("rflux_pagerank", out, "out", [&] {
        return rflux::percolab::pagerank(adjacency, n, damping, max_iter, tol);
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

rflux_status rflux_mst_clusters(const double* distance, size_t n,
                                double threshold, int32_t* labels,
                                size_t* n_components, size_t* largest_size) {
    return api_guard("rflux_mst_clusters", [&] {
        require_output(labels, "labels");
        require_output(n_components, "n_components");
        require_output(largest_size, "largest_size");
        const rflux::percolab::ComponentSummary summary =
            rflux::percolab::mst_clusters(distance, n, threshold);
        copy_out_i32(summary.labels, labels);
        *n_components = summary.n_components;
        *largest_size = summary.largest_size;
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

rflux_status rflux_bond_percolation_weighted(
    const double* adjacency, size_t n, const double* probabilities,
    size_t n_probabilities, size_t repetitions, uint64_t seed,
    double* giant_mean, double* giant_std, double* susceptibility) {
    return guarded_percolation(
        "rflux_bond_percolation_weighted", giant_mean, giant_std,
        susceptibility, [&] {
            return rflux::percolab::bond_percolation_weighted(
                adjacency, n, probabilities, n_probabilities, repetitions,
                seed);
        });
}

rflux_status rflux_site_percolation_weighted(
    const double* adjacency, size_t n, const double* node_weights,
    const double* probabilities, size_t n_probabilities, size_t repetitions,
    uint64_t seed, double* giant_mean, double* giant_std,
    double* susceptibility) {
    return guarded_percolation(
        "rflux_site_percolation_weighted", giant_mean, giant_std,
        susceptibility, [&] {
            return rflux::percolab::site_percolation_weighted(
                adjacency, n, node_weights, probabilities, n_probabilities,
                repetitions, seed);
        });
}

rflux_status rflux_percolation_scan(
    const double* adjacency, size_t n, const double* edge_weights,
    const double* node_weights, int32_t site, const double* probabilities,
    size_t n_probabilities, size_t repetitions, uint64_t seed,
    double* giant_mean, double* giant_std, double* susceptibility) {
    return guarded_percolation(
        "rflux_percolation_scan", giant_mean, giant_std, susceptibility, [&] {
            if (site != 0 && site != 1) {
                throw rflux::InvalidArgument("site must be 0 or 1");
            }
            rflux::percolab::OccupationWeights weights;
            weights.edge = edge_weights;
            weights.node = node_weights;
            return rflux::percolab::percolation_scan(
                adjacency, n,
                site != 0 ? rflux::percolab::PercolationKind::site
                          : rflux::percolab::PercolationKind::bond,
                weights, probabilities, n_probabilities, repetitions, seed);
        });
}

rflux_status rflux_cluster_size_distribution(
    const double* adjacency, size_t n, const double* edge_weights,
    const double* node_weights, int32_t site, double probability,
    size_t repetitions, uint64_t seed, double* out) {
    return api_guard("rflux_cluster_size_distribution", [&] {
        require_output(out, "out");
        if (site != 0 && site != 1) {
            throw rflux::InvalidArgument("site must be 0 or 1");
        }
        rflux::percolab::OccupationWeights weights;
        weights.edge = edge_weights;
        weights.node = node_weights;
        const std::vector<double> values =
            rflux::percolab::cluster_size_distribution(
                adjacency, n,
                site != 0 ? rflux::percolab::PercolationKind::site
                          : rflux::percolab::PercolationKind::bond,
                weights, probability, repetitions, seed);
        copy_out(values, out);
    });
}

rflux_status rflux_targeted_removal(
    const double* adjacency, size_t n, const uint32_t* order,
    size_t checkpoints, size_t* removed, double* giant_fraction,
    double* susceptibility, size_t* n_components) {
    return api_guard("rflux_targeted_removal", [&] {
        require_output(removed, "removed");
        require_output(giant_fraction, "giant_fraction");
        require_output(susceptibility, "susceptibility");
        require_output(n_components, "n_components");
        const std::vector<rflux::percolab::RemovalPoint> points =
            rflux::percolab::targeted_removal(adjacency, n, order,
                                              checkpoints);
        for (std::size_t i = 0; i < points.size(); ++i) {
            removed[i] = points[i].removed;
            giant_fraction[i] = points[i].giant_fraction;
            susceptibility[i] = points[i].finite_susceptibility;
            n_components[i] = points[i].n_components;
        }
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

rflux_status rflux_factor_momentum(const double* close, size_t n_obs,
                                   size_t n_assets, size_t window,
                                   double* out) {
    return guarded_vector("rflux_factor_momentum", out, "out", [&] {
        rflux::core::FactorContext context;
        context.close = {close, n_obs, n_assets};
        return rflux::factorlab::MomentumFactor(window).compute(context);
    });
}

rflux_status rflux_factor_volume_reversal(const double* close,
                                          const double* volume, size_t n_obs,
                                          size_t n_assets, size_t window,
                                          double* out) {
    return guarded_vector("rflux_factor_volume_reversal", out, "out", [&] {
        rflux::core::FactorContext context;
        context.close = {close, n_obs, n_assets};
        context.volume = {volume, n_obs, n_assets};
        return rflux::factorlab::VolumeReversalFactor(window).compute(context);
    });
}

rflux_status rflux_factor_vwap_deviation(const double* close,
                                         const double* volume, size_t n_obs,
                                         size_t n_assets, size_t window,
                                         double* out) {
    return guarded_vector("rflux_factor_vwap_deviation", out, "out", [&] {
        rflux::core::FactorContext context;
        context.close = {close, n_obs, n_assets};
        context.volume = {volume, n_obs, n_assets};
        return rflux::factorlab::VwapDeviationFactor(window).compute(context);
    });
}

rflux_status rflux_factor_ic(const double* factor,
                              const double* forward_returns, size_t n_obs,
                              size_t n_assets, int32_t spearman, double* out) {
    return guarded_vector("rflux_factor_ic", out, "out", [&] {
        return rflux::factorlab::information_coefficient(
            factor, forward_returns, n_obs, n_assets, spearman != 0);
    });
}

rflux_status rflux_factor_turnover(const double* weights, size_t n_obs,
                                   size_t n_assets, double* out) {
    return guarded_vector("rflux_factor_turnover", out, "out", [&] {
        return rflux::factorlab::factor_turnover(weights, n_obs, n_assets);
    });
}

rflux_status rflux_factor_amihud(const double* close, const double* volume,
                                 size_t n_obs, size_t n_assets, size_t window,
                                 double scale, double* out) {
    return guarded_vector("rflux_factor_amihud", out, "out", [&] {
        rflux::core::FactorContext context;
        context.close = {close, n_obs, n_assets};
        context.volume = {volume, n_obs, n_assets};
        return rflux::factorlab::AmihudFactor(window, scale).compute(context);
    });
}

rflux_status rflux_hrp_weights(const double* covariance, size_t n_assets,
                               double* weights) {
    return guarded_vector("rflux_hrp_weights", weights, "weights", [&] {
        return rflux::portfoliolab::HRPAllocator().weights(covariance,
                                                            n_assets);
    });
}

rflux_status rflux_hrp_order(const double* covariance, size_t n_assets,
                             size_t* order) {
    return api_guard("rflux_hrp_order", [&] {
        require_output(order, "order");
        const auto values =
            rflux::portfoliolab::quasi_diagonal_order(covariance, n_assets);
        std::copy(values.begin(), values.end(), order);
    });
}

rflux_status rflux_risk_parity_weights(const double* covariance,
                                       size_t n_assets, double* weights) {
    return guarded_vector("rflux_risk_parity_weights", weights, "weights",
                          [&] {
                              return rflux::portfoliolab::risk_parity_weights(
                                  covariance, n_assets);
                          });
}

rflux_status rflux_nco_weights(const double* covariance, size_t n_assets,
                               double distance_threshold, double* weights) {
    return guarded_vector("rflux_nco_weights", weights, "weights", [&] {
        return rflux::portfoliolab::nco_weights(
            covariance, n_assets, distance_threshold);
    });
}

rflux_status rflux_rebalance_targets(const double* current,
                                     const double* target, size_t n_rows,
                                     size_t n_assets, double threshold,
                                     double* out) {
    return guarded_vector("rflux_rebalance_targets", out, "out", [&] {
        return rflux::portfoliolab::rebalance_targets(
            current, target, n_rows, n_assets, threshold);
    });
}

rflux_status rflux_almgren_chriss_plan(
    double quantity, size_t slices, double horizon, double temporary_impact,
    double permanent_impact, double volatility, double risk_aversion,
    double* trades, double* expected_cost, double* expected_variance,
    double* kappa) {
    return api_guard("rflux_almgren_chriss_plan", [&] {
        require_output(trades, "trades");
        const bool wants_scalars =
            expected_cost != nullptr || expected_variance != nullptr ||
            kappa != nullptr;
        if (wants_scalars) {
            require_output(expected_cost, "expected_cost");
            require_output(expected_variance, "expected_variance");
            require_output(kappa, "kappa");
        }
        const rflux::core::ExecutionPlan plan =
            rflux::execution::AlmgrenChrissPlanner(
                temporary_impact, permanent_impact, volatility, risk_aversion)
                .plan(quantity, slices, horizon);
        copy_out(plan.trades, trades);
        if (wants_scalars) {
            *expected_cost = plan.expected_cost;
            *expected_variance = plan.expected_variance;
            *kappa = plan.kappa;
        }
    });
}

rflux_status rflux_tail_dependence(const double* returns, size_t n_obs,
                                   size_t n_assets, double q, int32_t upper,
                                   double* out) {
    return guarded_vector("rflux_tail_dependence", out, "out", [&] {
        return rflux::risk::tail_dependence(returns, n_obs, n_assets, q,
                                            upper != 0);
    });
}

rflux_status rflux_cvar_loss(const double* returns, size_t n, double alpha,
                             double* out) {
    return api_guard("rflux_cvar_loss", [&] {
        require_output(out, "out");
        *out = rflux::mllab::cvar_loss(returns, n, alpha);
    });
}

rflux_status rflux_graph_smooth(const double* signal,
                                const double* adjacency, size_t n,
                                double alpha, size_t steps, double* out) {
    return guarded_vector("rflux_graph_smooth", out, "out", [&] {
        return rflux::mllab::graph_smooth(signal, adjacency, n, alpha,
                                          steps);
    });
}

rflux_status rflux_order_book_metrics(
    const double* bid_price, const double* bid_size, const double* ask_price,
    const double* ask_size, size_t n, double* spread, double* mid,
    double* microprice, double* imbalance) {
    return api_guard("rflux_order_book_metrics", [&] {
        require_output(spread, "spread");
        require_output(mid, "mid");
        require_output(microprice, "microprice");
        require_output(imbalance, "imbalance");
        const auto values = rflux::execution::order_book_metrics(
            bid_price, bid_size, ask_price, ask_size, n);
        std::copy(values.spread.begin(), values.spread.end(), spread);
        std::copy(values.mid.begin(), values.mid.end(), mid);
        std::copy(values.microprice.begin(), values.microprice.end(),
                  microprice);
        std::copy(values.imbalance.begin(), values.imbalance.end(), imbalance);
    });
}

rflux_status rflux_huber_regression(
    const double* features, const double* target, size_t n_obs,
    size_t n_features, double epsilon, size_t max_iter, double tol,
    int32_t fit_intercept, double* coefficients, double* intercept,
    double* scale, size_t* iterations, int32_t* converged) {
    return api_guard("rflux_huber_regression", [&] {
        require_output(coefficients, "coefficients");
        require_output(intercept, "intercept");
        require_output(scale, "scale");
        require_output(iterations, "iterations");
        require_output(converged, "converged");
        if (fit_intercept != 0 && fit_intercept != 1) {
            throw rflux::InvalidArgument("fit_intercept must be 0 or 1");
        }
        rflux::mllab::HuberRegressor model(
            epsilon, max_iter, tol, fit_intercept != 0);
        model.fit(features, target, n_obs, n_features);
        copy_out(model.coefficients(), coefficients);
        *intercept = model.intercept();
        *scale = model.scale();
        *iterations = model.iterations();
        *converged = model.converged() ? 1 : 0;
    });
}

rflux_status rflux_site_percolation_analyze(
    const double* adjacency, size_t n, double p, size_t repetitions,
    uint64_t seed, double crisis_fraction, size_t threshold_points,
    double* giant_fraction, double* giant_std, double* susceptibility,
    size_t* largest_component, int32_t* systemic) {
    return api_guard("rflux_site_percolation_analyze", [&] {
        require_output(giant_fraction, "giant_fraction");
        require_output(giant_std, "giant_std");
        require_output(susceptibility, "susceptibility");
        require_output(largest_component, "largest_component");
        require_output(systemic, "systemic");
        rflux::percolab::SitePercolationDetector::Options options;
        options.repetitions = repetitions;
        options.seed = seed;
        options.crisis_fraction = crisis_fraction;
        options.threshold_points = threshold_points;
        rflux::percolab::SitePercolationDetector detector(options);
        const auto network = rflux::percolab::make_network(adjacency, n);
        const auto result = detector.analyze(network, p);
        *giant_fraction = result.giant_component_fraction;
        *giant_std = result.giant_component_std;
        *susceptibility = result.susceptibility;
        *largest_component = result.largest_component;
        *systemic = result.is_systemic_crisis ? 1 : 0;
    });
}

rflux_status rflux_site_percolation_threshold(
    const double* adjacency, size_t n, size_t repetitions, uint64_t seed,
    double crisis_fraction, size_t threshold_points, double* threshold) {
    return guarded_scalar(
        "rflux_site_percolation_threshold", threshold, "threshold", [&] {
            rflux::percolab::SitePercolationDetector::Options options;
            options.repetitions = repetitions;
            options.seed = seed;
            options.crisis_fraction = crisis_fraction;
            options.threshold_points = threshold_points;
            rflux::percolab::SitePercolationDetector detector(options);
            const auto network = rflux::percolab::make_network(adjacency, n);
            return detector.estimate_threshold(network);
        });
}

rflux_status rflux_pretrade_check(
    int64_t quantity, double price, int32_t side,
    double projected_position_fraction, double current_drawdown_fraction,
    double projected_turnover, double max_position_limit,
    double max_drawdown_limit, double turnover_threshold,
    int32_t systemic_crisis, int32_t* allowed, char* reason,
    size_t reason_capacity) {
    return api_guard("rflux_pretrade_check", [&] {
        require_output(allowed, "allowed");
        require_output(reason, "reason");
        if (reason_capacity == 0) {
            throw rflux::InvalidArgument("reason_capacity must be positive");
        }
        if (side != 0 && side != 1) {
            throw rflux::InvalidArgument("side must be 0 or 1");
        }
        if (systemic_crisis != 0 && systemic_crisis != 1) {
            throw rflux::InvalidArgument("systemic_crisis must be 0 or 1");
        }
        rflux::core::RiskConstraints constraints;
        constraints.max_position_limit = max_position_limit;
        constraints.max_drawdown_limit = max_drawdown_limit;
        constraints.turnover_threshold = turnover_threshold;
        rflux::core::Order order;
        order.quantity = quantity;
        order.price = price;
        order.side = side == 0 ? rflux::core::Side::buy
                               : rflux::core::Side::sell;
        rflux::percolab::PercolationResult crisis;
        crisis.is_systemic_crisis = systemic_crisis != 0;
        const auto decision = rflux::risk::PreTradeRiskControl(constraints)
                                  .check(order, projected_position_fraction,
                                         current_drawdown_fraction,
                                         projected_turnover, &crisis);
        *allowed = decision.allowed ? 1 : 0;
        std::snprintf(reason, reason_capacity, "%s",
                      decision.reason.c_str());
    });
}

}  // extern "C"
