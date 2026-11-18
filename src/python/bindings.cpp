/*
 * Native Python bridge for rflux.
 *
 * Design rules
 * ------------
 * - NumPy arrays are accepted through pybind11's buffer-aware array caster.
 *   Contiguous float64 inputs are passed to the C++ core without copying.
 * - Expensive kernels release the GIL for their whole C++ section.
 * - The module is built with pybind11's STABLE_ABI mode when supported by the
 *   build environment, so one binary serves every CPython >= 3.12.
 * - No NumPy C API is used directly; pybind11's numpy.h handles the buffer and
 *   dtype details while the limited API is active.
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "internal/common.hpp"
#include "rflux/rflux.h"
#include "rflux/execution/almgren_chriss.hpp"
#include "rflux/factorlab/backtest.hpp"
#include "rflux/factorlab/evaluation.hpp"
#include "rflux/factorlab/factors.hpp"
#include "rflux/mllab/huber.hpp"
#include "rflux/percolab/network.hpp"
#include "rflux/percolab/percolation.hpp"
#include "rflux/portfoliolab/advice.hpp"
#include "rflux/portfoliolab/hrp.hpp"
#include "rflux/portfoliolab/risk.hpp"
#include "rflux/qflux/indicators.hpp"
#include "rflux/qflux/performance.hpp"
#include "rflux/qflux/ranking.hpp"
#include "rflux/qflux/stats.hpp"
#include "rflux/qflux/window.hpp"
#include "rflux/regimelab/alignment.hpp"
#include "rflux/regimelab/attribution.hpp"
#include "rflux/regimelab/hmm.hpp"
#include "rflux/regimelab/regime.hpp"
#include "rflux/risk/tail_dependence.hpp"

namespace py = pybind11;

#ifndef RFLUX_VERSION_STRING
#define RFLUX_VERSION_STRING "0.0.0"
#endif

#ifndef RFLUX_NATIVE_STABLE_ABI
#define RFLUX_NATIVE_STABLE_ABI 0
#endif

namespace {

using DoubleArray =
    py::array_t<double, py::array::c_style | py::array::forcecast>;

PyObject* g_native_error = nullptr;

constexpr int kInvalidArgument = 1;
constexpr int kOutOfMemory = 2;
constexpr int kNonFinite = 3;
constexpr int kNotReady = 4;
constexpr int kNumerical = 5;

void set_native_error(int status, const char* function, const char* message) {
    if (g_native_error == nullptr) {
        PyErr_SetString(PyExc_RuntimeError, message);
        return;
    }
    py::object exc =
        py::reinterpret_borrow<py::object>(g_native_error)(py::str(message));
    exc.attr("status") = py::int_(status);
    exc.attr("function") = py::str(function);
    PyErr_SetObject(g_native_error, exc.ptr());
}

void register_error_translator() {
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const rflux::NotReady& e) {
            set_native_error(kNotReady, "rflux_native", e.what());
        } catch (const rflux::NonFinite& e) {
            set_native_error(kNonFinite, "rflux_native", e.what());
        } catch (const rflux::InvalidArgument& e) {
            set_native_error(kInvalidArgument, "rflux_native", e.what());
        } catch (const rflux::NumericalError& e) {
            set_native_error(kNumerical, "rflux_native", e.what());
        } catch (const std::bad_alloc& e) {
            set_native_error(kOutOfMemory, "rflux_native", e.what());
        }
    });
}

std::size_t as_size_arg(double value, const char* name,
                        std::size_t minimum = 1) {
    if (!std::isfinite(value) || value < static_cast<double>(minimum) ||
        std::floor(value) != value ||
        value > static_cast<double>(std::numeric_limits<std::size_t>::max())) {
        throw rflux::InvalidArgument(std::string(name) +
                                     " must be an integer >= " +
                                     std::to_string(minimum));
    }
    return static_cast<std::size_t>(value);
}

std::vector<double> parse_numeric_args(const py::args& args) {
    std::vector<double> result;
    result.reserve(args.size());
    for (const auto& item : args) {
        result.push_back(py::cast<double>(item));
    }
    return result;
}

void require_arg_count(const std::vector<double>& args, std::size_t expected,
                       const char* operation) {
    if (args.size() != expected) {
        throw rflux::InvalidArgument(std::string(operation) + " expects " +
                                     std::to_string(expected) +
                                     " numeric arguments");
    }
}

void require_1d(const DoubleArray& values, const char* name) {
    if (values.ndim() != 1) {
        throw rflux::InvalidArgument(std::string(name) +
                                     " must be one-dimensional");
    }
}

void require_2d(const DoubleArray& values, const char* name) {
    if (values.ndim() != 2) {
        throw rflux::InvalidArgument(std::string(name) +
                                     " must be two-dimensional");
    }
}

std::size_t matching_1d_length(const DoubleArray& x, const DoubleArray& y,
                               const char* x_name = "x",
                               const char* y_name = "y") {
    require_1d(x, x_name);
    require_1d(y, y_name);
    if (x.size() != y.size()) {
        throw rflux::InvalidArgument(std::string(x_name) + " and " + y_name +
                                     " must have equal lengths");
    }
    return static_cast<std::size_t>(x.shape(0));
}

/* One allocation-and-copy path for every NumPy result, whatever its rank. */
template <typename T>
py::array_t<T> array_from(const std::vector<T>& values,
                          std::initializer_list<py::ssize_t> shape) {
    py::array_t<T> out(shape);
    if (!values.empty()) {
        std::memcpy(out.mutable_data(), values.data(),
                    values.size() * sizeof(T));
    }
    return out;
}

py::array_t<double> vector_to_array(const std::vector<double>& values) {
    return array_from(values, {static_cast<py::ssize_t>(values.size())});
}

py::array_t<double> matrix_to_array(const std::vector<double>& values,
                                    py::ssize_t rows, py::ssize_t columns) {
    return array_from(values, {rows, columns});
}

py::array_t<std::int32_t> int_vector_to_array(
    const std::vector<std::int32_t>& values) {
    return array_from(values, {static_cast<py::ssize_t>(values.size())});
}

/*
 * Run one native section with the GIL released.  Centralising the release here
 * keeps every entry point on the same exception-safe reacquire path and
 * removes the repeated brace-scoped blocks.
 */
template <typename Function>
auto run_released(Function&& function) -> decltype(function()) {
    py::gil_scoped_release release;
    return function();
}

/* Allocate a NumPy result and let a kernel fill it while the GIL is free. */
template <typename T = double, typename Kernel>
py::array_t<T> filled_array(std::initializer_list<py::ssize_t> shape,
                            Kernel&& kernel) {
    py::array_t<T> result(shape);
    T* out = result.mutable_data();
    run_released([&] { kernel(out); });
    return result;
}

/* Three parallel series (MACD and Bollinger bands) share one tuple builder. */
template <typename Kernel>
py::tuple filled_triple(std::size_t n, Kernel&& kernel) {
    py::array_t<double> first(static_cast<py::ssize_t>(n));
    py::array_t<double> second(static_cast<py::ssize_t>(n));
    py::array_t<double> third(static_cast<py::ssize_t>(n));
    double* a = first.mutable_data();
    double* b = second.mutable_data();
    double* c = third.mutable_data();
    run_released([&] { kernel(a, b, c); });
    return py::make_tuple(first, second, third);
}

/*
 * Buffer-first dispatch shared by `series` and `series_into`.  The kernel
 * writes straight into the caller's storage, so neither entry point allocates
 * a temporary result vector.
 */
void run_series_into(const std::string& name, const double* values,
                     std::size_t n, const std::vector<double>& args,
                     double* out) {
    using namespace rflux::qflux;

    if (name == "rflux_returns") {
        require_arg_count(args, 1, "returns");
        returns_into(values, n, args[0] != 0.0, out);
        return;
    }
    if (name == "rflux_sma") {
        require_arg_count(args, 1, "sma");
        sma_into(values, n, as_size_arg(args[0], "window"), out);
        return;
    }
    if (name == "rflux_rolling_sum") {
        require_arg_count(args, 1, "rolling_sum");
        rolling_sum_into(values, n, as_size_arg(args[0], "window"), out);
        return;
    }
    if (name == "rflux_rolling_std") {
        require_arg_count(args, 2, "rolling_std");
        rolling_std_into(values, n, as_size_arg(args[0], "window"),
                         as_size_arg(args[1], "ddof", 0), out);
        return;
    }
    if (name == "rflux_rolling_min") {
        require_arg_count(args, 1, "rolling_min");
        rolling_min_into(values, n, as_size_arg(args[0], "window"), out);
        return;
    }
    if (name == "rflux_rolling_max") {
        require_arg_count(args, 1, "rolling_max");
        rolling_max_into(values, n, as_size_arg(args[0], "window"), out);
        return;
    }
    if (name == "rflux_rolling_zscore") {
        require_arg_count(args, 2, "rolling_zscore");
        rolling_zscore_into(values, n, as_size_arg(args[0], "window"),
                            as_size_arg(args[1], "ddof", 0), out);
        return;
    }
    if (name == "rflux_rolling_quantile") {
        require_arg_count(args, 2, "rolling_quantile");
        rolling_quantile_into(values, n, as_size_arg(args[0], "window"),
                              args[1], out);
        return;
    }
    if (name == "rflux_ema") {
        require_arg_count(args, 1, "ema");
        ema_into(values, n, as_size_arg(args[0], "span"), out);
        return;
    }
    if (name == "rflux_rsi") {
        require_arg_count(args, 1, "rsi");
        rsi_into(values, n, as_size_arg(args[0], "period"), out);
        return;
    }
    if (name == "rflux_drawdown") {
        require_arg_count(args, 0, "drawdown");
        drawdown_into(values, n, out);
        return;
    }
    if (name == "rflux_rolling_slope") {
        require_arg_count(args, 1, "rolling_slope");
        rolling_slope_into(values, n, as_size_arg(args[0], "window"), out);
        return;
    }
    if (name == "rflux_rolling_r2") {
        require_arg_count(args, 1, "rolling_r2");
        rolling_r2_into(values, n, as_size_arg(args[0], "window"), out);
        return;
    }
    if (name == "rflux_rank") {
        require_arg_count(args, 0, "rank");
        rank_into(values, n, out);
        return;
    }
    if (name == "rflux_percentile_rank") {
        require_arg_count(args, 0, "percentile_rank");
        percentile_rank_into(values, n, out);
        return;
    }
    if (name == "rflux_zscore") {
        require_arg_count(args, 1, "zscore");
        zscore_into(values, n, as_size_arg(args[0], "ddof", 0), out);
        return;
    }
    if (name == "rflux_winsorize") {
        require_arg_count(args, 2, "winsorize");
        winsorize_into(values, n, args[0], args[1], out);
        return;
    }
    throw rflux::InvalidArgument("unsupported native series operation: " + name);
}

void run_pair_series_into(const std::string& name, const double* x,
                          const double* y, std::size_t n,
                          const std::vector<double>& args, double* out) {
    using namespace rflux::qflux;

    if (name == "rflux_rolling_corr") {
        require_arg_count(args, 1, "rolling_corr");
        rolling_corr_into(x, y, n, as_size_arg(args[0], "window"), out);
        return;
    }
    if (name == "rflux_rolling_cov") {
        require_arg_count(args, 2, "rolling_cov");
        rolling_cov_into(x, y, n, as_size_arg(args[0], "window"),
                         as_size_arg(args[1], "ddof", 0), out);
        return;
    }
    if (name == "rflux_rolling_beta") {
        require_arg_count(args, 1, "rolling_beta");
        rolling_beta_into(x, y, n, as_size_arg(args[0], "window"), out);
        return;
    }
    throw rflux::InvalidArgument("unsupported native pair operation: " + name);
}

py::array_t<double> series(const std::string& name, const DoubleArray& values,
                           const py::args& args) {
    require_1d(values, "values");
    const auto n = static_cast<std::size_t>(values.shape(0));
    const std::vector<double> numeric_args = parse_numeric_args(args);
    return filled_array({static_cast<py::ssize_t>(n)}, [&](double* out) {
        run_series_into(name, values.data(), n, numeric_args, out);
    });
}

py::array series_into(const std::string& name, const DoubleArray& values,
                      py::array out, const py::args& args) {
    require_1d(values, "values");
    const auto n = static_cast<std::size_t>(values.shape(0));
    const auto info = out.request();
    if (info.ndim != 1 || static_cast<std::size_t>(info.shape[0]) != n) {
        throw rflux::InvalidArgument(
            "out must be a one-dimensional array with the same length as values");
    }
    if (info.format != py::format_descriptor<double>::format() ||
        info.itemsize != static_cast<py::ssize_t>(sizeof(double)) ||
        info.strides[0] != static_cast<py::ssize_t>(sizeof(double))) {
        throw rflux::InvalidArgument(
            "out must be a contiguous float64 array (zero-copy requirement)");
    }
    if (!out.writeable()) {
        throw rflux::InvalidArgument("out must be writeable");
    }
    const std::vector<double> numeric_args = parse_numeric_args(args);
    double* target = static_cast<double*>(info.ptr);
    run_released([&] {
        run_series_into(name, values.data(), n, numeric_args, target);
    });
    return out;
}

py::array_t<double> pair_series(const std::string& name, const DoubleArray& x,
                                const DoubleArray& y, const py::args& args) {
    const auto n = matching_1d_length(x, y);
    const std::vector<double> numeric_args = parse_numeric_args(args);
    return filled_array({static_cast<py::ssize_t>(n)}, [&](double* out) {
        run_pair_series_into(name, x.data(), y.data(), n, numeric_args, out);
    });
}

py::array_t<double> atr_series(const DoubleArray& high, const DoubleArray& low,
                               const DoubleArray& close, double period) {
    require_1d(high, "high");
    require_1d(low, "low");
    require_1d(close, "close");
    if (high.size() != low.size() || high.size() != close.size()) {
        throw rflux::InvalidArgument(
            "high, low and close must have equal lengths");
    }
    const auto n = static_cast<std::size_t>(high.shape(0));
    const std::size_t period_size = as_size_arg(period, "period");
    return filled_array({static_cast<py::ssize_t>(n)}, [&](double* out) {
        rflux::qflux::atr_into(high.data(), low.data(), close.data(), n,
                               period_size, out);
    });
}

py::tuple macd_result(const DoubleArray& prices, std::size_t fast,
                       std::size_t slow, std::size_t signal) {
    require_1d(prices, "prices");
    const auto n = static_cast<std::size_t>(prices.shape(0));
    return filled_triple(n, [&](double* macd, double* signal_line,
                                double* histogram) {
        rflux::qflux::macd_into(prices.data(), n, fast, slow, signal, macd,
                                signal_line, histogram);
    });
}

py::tuple bollinger_result(const DoubleArray& prices, std::size_t window,
                           double n_std) {
    require_1d(prices, "prices");
    const auto n = static_cast<std::size_t>(prices.shape(0));
    return filled_triple(n, [&](double* middle, double* upper, double* lower) {
        rflux::qflux::bollinger_into(prices.data(), n, window, n_std, middle,
                                     upper, lower);
    });
}

py::tuple describe_values(const DoubleArray& values) {
    require_1d(values, "values");
    const auto n = static_cast<std::size_t>(values.shape(0));
    const rflux::qflux::Moments result = run_released([&] {
        return rflux::qflux::describe(values.data(), n);
    });
    return py::make_tuple(result.mean, result.variance, result.stddev,
                          result.skewness, result.excess_kurtosis,
                          result.n_valid);
}

double quantile_value(const DoubleArray& values, double q) {
    require_1d(values, "values");
    const auto n = static_cast<std::size_t>(values.shape(0));
    return run_released(
        [&] { return rflux::qflux::quantile(values.data(), n, q); });
}

double pearson_value(const DoubleArray& x, const DoubleArray& y) {
    const auto n = matching_1d_length(x, y);
    return run_released(
        [&] { return rflux::qflux::pearson(x.data(), y.data(), n); });
}

py::array_t<double> correlation_matrix_values(const DoubleArray& panel) {
    require_2d(panel, "panel");
    const auto n_obs = static_cast<std::size_t>(panel.shape(0));
    const auto n_assets = static_cast<std::size_t>(panel.shape(1));
    return filled_array(
        {static_cast<py::ssize_t>(n_assets),
         static_cast<py::ssize_t>(n_assets)},
        [&](double* out) {
            rflux::qflux::correlation_matrix_into(panel.data(), n_obs,
                                                  n_assets, out);
        });
}

using PerformanceRatio = double (*)(const double*, std::size_t, double, double);

double ratio_value(const DoubleArray& values, PerformanceRatio kernel,
                   double periods_per_year, double risk_free) {
    require_1d(values, "returns");
    const auto n = static_cast<std::size_t>(values.shape(0));
    return run_released([&] {
        return kernel(values.data(), n, periods_per_year, risk_free);
    });
}

double sharpe_value(const DoubleArray& values, double periods_per_year,
                    double risk_free) {
    return ratio_value(values, &rflux::qflux::sharpe, periods_per_year,
                       risk_free);
}

double sortino_value(const DoubleArray& values, double periods_per_year,
                     double risk_free) {
    return ratio_value(values, &rflux::qflux::sortino, periods_per_year,
                       risk_free);
}

py::array_t<double> equity_curve(const DoubleArray& returns,
                                  double initial) {
    require_1d(returns, "returns");
    std::vector<double> result = run_released([&] {
        return rflux::qflux::equity_curve(
            returns.data(), static_cast<std::size_t>(returns.shape(0)),
            initial);
    });
    return vector_to_array(result);
}

py::tuple performance_summary(const DoubleArray& returns,
                             double periods_per_year, double risk_free,
                             double confidence, std::size_t periods) {
    require_1d(returns, "returns");
    const auto summary = run_released([&] {
        return rflux::qflux::performance_summary(
            returns.data(), static_cast<std::size_t>(returns.shape(0)),
            periods_per_year, risk_free, confidence, periods);
    });
    std::vector<double> values = {
        summary.total_return,          summary.annualized_return,
        summary.annualized_volatility, summary.sharpe,
        summary.sortino,               summary.max_drawdown,
        summary.calmar,                summary.var,
        summary.cvar,                  summary.hit_rate};
    return py::make_tuple(summary.n_periods, summary.elapsed_periods,
                          vector_to_array(values));
}

py::tuple var_cvar_value(const DoubleArray& values, double confidence) {
    require_1d(values, "returns");
    const auto n = static_cast<std::size_t>(values.shape(0));
    const rflux::qflux::TailRisk result = run_released([&] {
        return rflux::qflux::historical_var_cvar(values.data(), n, confidence);
    });
    return py::make_tuple(result.var, result.cvar);
}

py::array_t<double> neutralize_values(
    const DoubleArray& values,
    const py::array_t<std::int32_t, py::array::c_style | py::array::forcecast>&
        groups,
    std::size_t group_count) {
    require_1d(values, "values");
    if (groups.ndim() != 1 || groups.size() != values.size()) {
        throw rflux::InvalidArgument(
            "groups must be one-dimensional and match values");
    }
    const auto n = static_cast<std::size_t>(values.shape(0));
    return filled_array({static_cast<py::ssize_t>(n)}, [&](double* out) {
        rflux::qflux::neutralize_into(values.data(), n, groups.data(),
                                      group_count, out);
    });
}

rflux::core::FactorContext make_factor_context(
    const DoubleArray& close, const DoubleArray* volume = nullptr) {
    require_2d(close, "close");
    rflux::core::FactorContext context;
    context.close = {close.data(),
                     static_cast<std::size_t>(close.shape(0)),
                     static_cast<std::size_t>(close.shape(1))};
    if (volume != nullptr) {
        require_2d(*volume, "volume");
        if (close.shape(0) != volume->shape(0) ||
            close.shape(1) != volume->shape(1)) {
            throw rflux::InvalidArgument("close and volume panels must match");
        }
        context.volume = {volume->data(),
                          static_cast<std::size_t>(volume->shape(0)),
                          static_cast<std::size_t>(volume->shape(1))};
    }
    return context;
}

template <typename Compute>
py::array_t<double> compute_factor_panel(const DoubleArray& close,
                                         const DoubleArray* volume,
                                         Compute&& compute) {
    const rflux::core::FactorContext context =
        make_factor_context(close, volume);
    std::vector<double> result = run_released([&] { return compute(context); });
    return matrix_to_array(result, close.shape(0), close.shape(1));
}

template <typename Factory>
py::array_t<double> window_factor(const DoubleArray& close,
                                  const DoubleArray* volume,
                                  Factory&& make_factor) {
    return compute_factor_panel(
        close, volume,
        [&](const rflux::core::FactorContext& context) {
            return make_factor().compute(context);
        });
}

py::array_t<double> factor_momentum(const DoubleArray& close,
                                    std::size_t window) {
    return window_factor(close, nullptr, [window] {
        return rflux::factorlab::MomentumFactor(window);
    });
}

py::array_t<double> factor_volume_reversal(const DoubleArray& close,
                                           const DoubleArray& volume,
                                           std::size_t window) {
    return window_factor(close, &volume, [window] {
        return rflux::factorlab::VolumeReversalFactor(window);
    });
}

py::array_t<double> factor_vwap_deviation(const DoubleArray& close,
                                          const DoubleArray& volume,
                                          std::size_t window) {
    return window_factor(close, &volume, [window] {
        return rflux::factorlab::VwapDeviationFactor(window);
    });
}

py::array_t<double> factor_ic(const DoubleArray& factor,
                              const DoubleArray& forward_returns,
                              bool spearman, std::size_t min_pairs) {
    require_2d(factor, "factor");
    require_2d(forward_returns, "forward_returns");
    if (factor.shape(0) != forward_returns.shape(0) ||
        factor.shape(1) != forward_returns.shape(1)) {
        throw rflux::InvalidArgument(
            "factor and forward_returns must have equal shape");
    }
    std::vector<double> result = run_released([&] {
        return rflux::factorlab::information_coefficient(
            factor.data(), forward_returns.data(),
            static_cast<std::size_t>(factor.shape(0)),
            static_cast<std::size_t>(factor.shape(1)), spearman, min_pairs);
    });
    return vector_to_array(result);
}

py::array_t<double> factor_turnover(const DoubleArray& weights) {
    require_2d(weights, "weights");
    std::vector<double> result = run_released([&] {
        return rflux::factorlab::factor_turnover(
            weights.data(), static_cast<std::size_t>(weights.shape(0)),
            static_cast<std::size_t>(weights.shape(1)));
    });
    return vector_to_array(result);
}

py::array_t<double> factor_amihud(const DoubleArray& close,
                                  const DoubleArray& volume,
                                  std::size_t window, double scale) {
    return window_factor(close, &volume, [window, scale] {
        return rflux::factorlab::AmihudFactor(window, scale);
    });
}

py::array_t<double> factor_forward_returns(const DoubleArray& prices,
                                           std::size_t periods, bool log) {
    require_2d(prices, "prices");
    const auto n_obs = static_cast<std::size_t>(prices.shape(0));
    const auto n_assets = static_cast<std::size_t>(prices.shape(1));
    std::vector<double> result = run_released([&] {
        return rflux::factorlab::forward_returns(prices.data(), n_obs,
                                                 n_assets, periods, log);
    });
    return matrix_to_array(result, prices.shape(0), prices.shape(1));
}

py::array_t<double> factor_quantile_returns(
    const DoubleArray& factor, const DoubleArray& forward_returns,
    std::size_t n_quantiles) {
    require_2d(factor, "factor");
    require_2d(forward_returns, "forward_returns");
    if (factor.shape(0) != forward_returns.shape(0) ||
        factor.shape(1) != forward_returns.shape(1)) {
        throw rflux::InvalidArgument(
            "factor and forward_returns must have equal shape");
    }
    const auto n_obs = static_cast<std::size_t>(factor.shape(0));
    std::vector<double> result = run_released([&] {
        return rflux::factorlab::quantile_portfolio_returns(
            factor.data(), forward_returns.data(), n_obs,
            static_cast<std::size_t>(factor.shape(1)), n_quantiles);
    });
    return matrix_to_array(result,
                           static_cast<py::ssize_t>(n_quantiles),
                           factor.shape(0));
}

py::tuple factor_long_short(const DoubleArray& factor,
                            const DoubleArray& forward_returns,
                            std::size_t n_quantiles, double cost_bps) {
    require_2d(factor, "factor");
    require_2d(forward_returns, "forward_returns");
    if (factor.shape(0) != forward_returns.shape(0) ||
        factor.shape(1) != forward_returns.shape(1)) {
        throw rflux::InvalidArgument(
            "factor and forward_returns must have equal shape");
    }
    const auto n_obs = static_cast<std::size_t>(factor.shape(0));
    const auto result = run_released([&] {
        return rflux::factorlab::long_short_backtest(
            factor.data(), forward_returns.data(), n_obs,
            static_cast<std::size_t>(factor.shape(1)), n_quantiles,
            cost_bps);
    });
    return py::make_tuple(
        vector_to_array(result.gross_returns), vector_to_array(result.costs),
        vector_to_array(result.turnover), vector_to_array(result.net_returns),
        vector_to_array(result.equity),
        matrix_to_array(result.quantile_returns,
                        static_cast<py::ssize_t>(n_quantiles),
                        factor.shape(0)));
}

py::tuple ic_summary(const DoubleArray& ic) {
    require_1d(ic, "ic");
    const auto summary = run_released([&] {
        return rflux::factorlab::summarize_ic(
            ic.data(), static_cast<std::size_t>(ic.shape(0)));
    });
    return py::make_tuple(summary.mean, summary.std,
                          summary.information_ratio, summary.positive_ratio,
                          summary.periods);
}

py::array_t<std::int32_t> regime_match_states(
    const DoubleArray& reference_means, const DoubleArray& reference_variances,
    const DoubleArray& target_means, const DoubleArray& target_variances,
    double variance_weight) {
    require_2d(reference_means, "reference_means");
    require_2d(reference_variances, "reference_variances");
    require_2d(target_means, "target_means");
    require_2d(target_variances, "target_variances");
    if (reference_means.shape(0) != reference_variances.shape(0) ||
        reference_means.shape(1) != reference_variances.shape(1) ||
        reference_means.shape(0) != target_means.shape(0) ||
        reference_means.shape(1) != target_means.shape(1) ||
        target_means.shape(0) != target_variances.shape(0) ||
        target_means.shape(1) != target_variances.shape(1)) {
        throw rflux::InvalidArgument(
            "state parameter panels must have equal shape");
    }
    std::vector<std::int32_t> result = run_released([&] {
        return rflux::regimelab::match_states(
            reference_means.data(), reference_variances.data(),
            target_means.data(), target_variances.data(),
            static_cast<std::size_t>(reference_means.shape(0)),
            static_cast<std::size_t>(reference_means.shape(1)),
            variance_weight);
    });
    return int_vector_to_array(result);
}

py::array_t<double> tail_dependence(const DoubleArray& returns, double q,
                                    bool upper) {
    require_2d(returns, "returns");
    std::vector<double> result = run_released([&] {
        return rflux::risk::tail_dependence(
            returns.data(), static_cast<std::size_t>(returns.shape(0)),
            static_cast<std::size_t>(returns.shape(1)), q, upper);
    });
    return matrix_to_array(result, returns.shape(1), returns.shape(1));
}

py::tuple portfolio_risk(const DoubleArray& returns,
                        const DoubleArray& weights, std::size_t ddof,
                        double periods_per_year) {
    require_2d(returns, "returns");
    require_1d(weights, "weights");
    if (returns.shape(1) != weights.shape(0)) {
        throw rflux::InvalidArgument(
            "weights must have one entry per asset");
    }
    const auto result = run_released([&] {
        return rflux::portfoliolab::portfolio_risk(
            returns.data(), static_cast<std::size_t>(returns.shape(0)),
            static_cast<std::size_t>(returns.shape(1)), weights.data(), ddof,
            periods_per_year);
    });
    return py::make_tuple(
        matrix_to_array(result.covariance, returns.shape(1),
                        returns.shape(1)),
        vector_to_array(result.asset_volatility),
        vector_to_array(result.component_volatility),
        vector_to_array(result.percent_contribution),
        vector_to_array(result.portfolio_returns),
        result.complete_observations, result.annualized_volatility,
        result.diversification_ratio, result.gross_hhi);
}

py::array_t<std::int64_t> hrp_order(const DoubleArray& covariance) {
    require_2d(covariance, "covariance");
    if (covariance.shape(0) != covariance.shape(1)) {
        throw rflux::InvalidArgument("covariance must be square");
    }
    const std::vector<std::size_t> order = run_released([&] {
        return rflux::portfoliolab::quasi_diagonal_order(
            covariance.data(), static_cast<std::size_t>(covariance.shape(0)));
    });
    std::vector<std::int64_t> converted(order.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        converted[i] = static_cast<std::int64_t>(order[i]);
    }
    return array_from(converted,
                      {static_cast<py::ssize_t>(converted.size())});
}

py::array_t<double> hrp_weights(const DoubleArray& covariance) {
    require_2d(covariance, "covariance");
    if (covariance.shape(0) != covariance.shape(1)) {
        throw rflux::InvalidArgument("covariance must be square");
    }
    std::vector<double> result = run_released([&] {
        return rflux::portfoliolab::HRPAllocator().weights(
            covariance.data(), static_cast<std::size_t>(covariance.shape(0)));
    });
    return vector_to_array(result);
}

py::array_t<double> risk_parity_weights(const DoubleArray& covariance) {
    require_2d(covariance, "covariance");
    if (covariance.shape(0) != covariance.shape(1)) {
        throw rflux::InvalidArgument("covariance must be square");
    }
    std::vector<double> result = run_released([&] {
        return rflux::portfoliolab::risk_parity_weights(
            covariance.data(), static_cast<std::size_t>(covariance.shape(0)));
    });
    return vector_to_array(result);
}

py::array_t<double> rebalance_targets(const DoubleArray& current,
                                      const DoubleArray& target,
                                      double threshold) {
    require_2d(current, "current");
    require_2d(target, "target");
    if (current.shape(0) != target.shape(0) ||
        current.shape(1) != target.shape(1)) {
        throw rflux::InvalidArgument("current and target must have equal shape");
    }
    std::vector<double> result = run_released([&] {
        return rflux::portfoliolab::rebalance_targets(
            current.data(), target.data(),
            static_cast<std::size_t>(current.shape(0)),
            static_cast<std::size_t>(current.shape(1)), threshold);
    });
    return matrix_to_array(result, current.shape(0), current.shape(1));
}

py::array_t<double> nco_weights(const DoubleArray& covariance,
                                double distance_threshold) {
    require_2d(covariance, "covariance");
    if (covariance.shape(0) != covariance.shape(1)) {
        throw rflux::InvalidArgument("covariance must be square");
    }
    std::vector<double> result = run_released([&] {
        return rflux::portfoliolab::nco_weights(
            covariance.data(), static_cast<std::size_t>(covariance.shape(0)),
            distance_threshold);
    });
    return vector_to_array(result);
}

py::tuple almgren_chriss(double quantity, std::size_t slices, double horizon,
                         double temporary_impact, double permanent_impact,
                         double volatility, double risk_aversion) {
    rflux::execution::AlmgrenChrissPlanner planner(
        temporary_impact, permanent_impact, volatility, risk_aversion);
    const rflux::core::ExecutionPlan plan = run_released([&] {
        return planner.plan(quantity, slices, horizon);
    });
    return py::make_tuple(vector_to_array(plan.trades), plan.expected_cost,
                          plan.expected_variance, plan.kappa);
}

py::array_t<double> linear_predict(const DoubleArray& features,
                                   const DoubleArray& coefficients,
                                   double intercept) {
    require_2d(features, "features");
    require_1d(coefficients, "coefficients");
    if (features.shape(1) != coefficients.shape(0)) {
        throw rflux::InvalidArgument(
            "features have the wrong number of columns");
    }
    std::vector<double> result = run_released([&] {
        return rflux::mllab::linear_predict(
            features.data(), static_cast<std::size_t>(features.shape(0)),
            static_cast<std::size_t>(features.shape(1)),
            coefficients.data(), intercept);
    });
    return vector_to_array(result);
}

double cvar_loss(const DoubleArray& returns, double alpha) {
    require_1d(returns, "returns");
    return run_released([&] {
        return rflux::mllab::cvar_loss(
            returns.data(), static_cast<std::size_t>(returns.shape(0)), alpha);
    });
}

py::array_t<double> graph_smooth(const DoubleArray& signal,
                                 const DoubleArray& adjacency, double alpha,
                                 std::size_t steps) {
    require_1d(signal, "signal");
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1) ||
        adjacency.shape(0) != signal.shape(0)) {
        throw rflux::InvalidArgument(
            "adjacency must be square and match signal length");
    }
    std::vector<double> result = run_released([&] {
        return rflux::mllab::graph_smooth(
            signal.data(), adjacency.data(),
            static_cast<std::size_t>(signal.shape(0)), alpha, steps);
    });
    return vector_to_array(result);
}

py::tuple order_book_metrics(const DoubleArray& bid_price,
                             const DoubleArray& bid_size,
                             const DoubleArray& ask_price,
                             const DoubleArray& ask_size) {
    require_1d(bid_price, "bid_price");
    require_1d(bid_size, "bid_size");
    require_1d(ask_price, "ask_price");
    require_1d(ask_size, "ask_size");
    if (bid_price.shape(0) != bid_size.shape(0) ||
        bid_price.shape(0) != ask_price.shape(0) ||
        bid_price.shape(0) != ask_size.shape(0)) {
        throw rflux::InvalidArgument(
            "order-book arrays must have equal lengths");
    }
    const auto result = run_released([&] {
        return rflux::execution::order_book_metrics(
            bid_price.data(), bid_size.data(), ask_price.data(),
            ask_size.data(), static_cast<std::size_t>(bid_price.shape(0)));
    });
    return py::make_tuple(vector_to_array(result.spread),
                          vector_to_array(result.mid),
                          vector_to_array(result.microprice),
                          vector_to_array(result.imbalance));
}

py::tuple huber_regression(const DoubleArray& features,
                           const DoubleArray& target, double epsilon,
                           std::size_t max_iter, double tol,
                           bool fit_intercept) {
    require_2d(features, "features");
    require_1d(target, "target");
    if (features.shape(0) != target.shape(0)) {
        throw rflux::InvalidArgument(
            "features and target must have the same row count");
    }
    rflux::mllab::HuberRegressor regressor(epsilon, max_iter, tol,
                                           fit_intercept);
    run_released([&] {
        regressor.fit(features.data(), target.data(),
                      static_cast<std::size_t>(features.shape(0)),
                      static_cast<std::size_t>(features.shape(1)));
    });
    return py::make_tuple(vector_to_array(regressor.coefficients()),
                          regressor.intercept(), regressor.scale(),
                          regressor.iterations(), regressor.converged());
}

class NativeHMM {
public:
    NativeHMM(std::size_t n_states, std::size_t n_features)
        : model_(std::make_shared<rflux::regimelab::GaussianHMM>(
              n_states, n_features)) {}

    NativeHMM& fit(const DoubleArray& observations, std::size_t max_iter,
                   std::size_t n_init, double tol, double variance_floor,
                   std::uint64_t seed, bool sort_states) {
        require_2d(observations, "observations");
        if (static_cast<std::size_t>(observations.shape(1)) !=
            model_->n_features()) {
            throw rflux::InvalidArgument(
                "observations have the wrong number of features");
        }
        rflux::regimelab::HmmOptions options;
        options.max_iter = max_iter;
        options.n_init = n_init;
        options.tol = tol;
        options.variance_floor = variance_floor;
        options.seed = seed;
        options.sort_states = sort_states;
        const auto n_obs = static_cast<std::size_t>(observations.shape(0));
        run_released([&] { model_->fit(observations.data(), n_obs, options); });
        return *this;
    }

    py::tuple fit_info() const {
        const auto iterations = model_->iterations();
        const auto converged = model_->converged();
        const auto log_likelihood = model_->log_likelihood();
        const auto aic = model_->aic();
        const auto bic = model_->bic();
        return py::make_tuple(iterations, converged, log_likelihood, aic, bic);
    }

    double score(const DoubleArray& observations) const {
        require_2d(observations, "observations");
        const auto n_obs = static_cast<std::size_t>(observations.shape(0));
        return run_released(
            [&] { return model_->score(observations.data(), n_obs); });
    }

    py::tuple parameters() const {
        const auto parameters = model_->parameters();
        const auto k = model_->n_states();
        const auto d = model_->n_features();
        return py::make_tuple(vector_to_array(parameters.start_prob),
                              matrix_to_array(parameters.transition,
                                              static_cast<py::ssize_t>(k),
                                              static_cast<py::ssize_t>(k)),
                              matrix_to_array(parameters.means,
                                              static_cast<py::ssize_t>(k),
                                              static_cast<py::ssize_t>(d)),
                              matrix_to_array(parameters.variances,
                                              static_cast<py::ssize_t>(k),
                                              static_cast<py::ssize_t>(d)));
    }

    void set_parameters(const DoubleArray& start_prob,
                        const DoubleArray& transition, const DoubleArray& means,
                        const DoubleArray& variances) {
        const auto k = model_->n_states();
        const auto d = model_->n_features();
        if (start_prob.size() != static_cast<py::ssize_t>(k)) {
            throw rflux::InvalidArgument("start_prob has the wrong length");
        }
        if (transition.size() != static_cast<py::ssize_t>(k * k)) {
            throw rflux::InvalidArgument("transition has the wrong shape");
        }
        if (means.size() != static_cast<py::ssize_t>(k * d) ||
            variances.size() != static_cast<py::ssize_t>(k * d)) {
            throw rflux::InvalidArgument(
                "means and variances have the wrong shape");
        }
        rflux::regimelab::HmmParameters parameters;
        parameters.start_prob.assign(start_prob.data(),
                                     start_prob.data() + start_prob.size());
        parameters.transition.assign(transition.data(),
                                     transition.data() + transition.size());
        parameters.means.assign(means.data(), means.data() + means.size());
        parameters.variances.assign(variances.data(),
                                    variances.data() + variances.size());
        run_released([&] { model_->set_parameters(parameters); });
    }

    template <typename Function>
    py::array_t<double> observations_matrix(const DoubleArray& observations,
                                            Function&& function) const {
        require_2d(observations, "observations");
        const auto n_obs = static_cast<std::size_t>(observations.shape(0));
        return matrix_to_array(
            run_released([&] { return function(observations.data(), n_obs); }),
            static_cast<py::ssize_t>(n_obs),
            static_cast<py::ssize_t>(model_->n_states()));
    }

    py::array_t<std::int32_t> viterbi(const DoubleArray& observations) const {
        require_2d(observations, "observations");
        const auto n_obs = static_cast<std::size_t>(observations.shape(0));
        return int_vector_to_array(run_released([&] {
            return model_->viterbi(observations.data(), n_obs);
        }));
    }

    py::array_t<double> filtered(const DoubleArray& observations) const {
        return observations_matrix(observations, [&](const double* data,
                                                     std::size_t n_obs) {
            return model_->filtered(data, n_obs);
        });
    }

    py::array_t<double> smoothed(const DoubleArray& observations) const {
        return observations_matrix(observations, [&](const double* data,
                                                     std::size_t n_obs) {
            return model_->smoothed(data, n_obs);
        });
    }

    py::array_t<double> forecast(const DoubleArray& state_probs,
                                 std::size_t horizon) const {
        require_1d(state_probs, "state_probs");
        const auto k = model_->n_states();
        if (state_probs.size() != static_cast<py::ssize_t>(k)) {
            throw rflux::InvalidArgument("state_probs has the wrong length");
        }
        return vector_to_array(run_released([&] {
            return model_->forecast(state_probs.data(), k, horizon);
        }));
    }

    py::array_t<double> filter_step(const DoubleArray& state_probs,
                                    const DoubleArray& observation) const {
        require_1d(state_probs, "state_probs");
        const auto k = model_->n_states();
        const auto d = model_->n_features();
        if (state_probs.size() != static_cast<py::ssize_t>(k)) {
            throw rflux::InvalidArgument("state_probs has the wrong length");
        }
        if (observation.size() != static_cast<py::ssize_t>(d)) {
            throw rflux::InvalidArgument("observation has the wrong length");
        }
        return vector_to_array(run_released([&] {
            return model_->filter_step(state_probs.data(), observation.data());
        }));
    }

    py::array_t<double> stationary_distribution() const {
        return vector_to_array(
            run_released([&] { return model_->stationary_distribution(); }));
    }

    py::array_t<double> expected_durations() const {
        return vector_to_array(
            run_released([&] { return model_->expected_durations(); }));
    }

    std::size_t n_states() const { return model_->n_states(); }
    std::size_t n_features() const { return model_->n_features(); }
    bool ready() const { return model_->ready(); }

private:
    std::shared_ptr<rflux::regimelab::GaussianHMM> model_;
};


py::array_t<double> equal_weight_market_returns(const DoubleArray& returns) {
    require_2d(returns, "returns");
    const auto n_obs = static_cast<std::size_t>(returns.shape(0));
    const auto n_assets = static_cast<std::size_t>(returns.shape(1));
    std::vector<double> result = run_released([&] {
        return rflux::qflux::equal_weight_market_returns(
            returns.data(), n_obs, n_assets);
    });
    return vector_to_array(result);
}

py::tuple finite_stats(const DoubleArray& values) {
    if (values.ndim() < 1 || values.ndim() > 2) {
        throw rflux::InvalidArgument("values must be 1D or 2D");
    }
    const auto stats = run_released([&] {
        return rflux::qflux::finite_stats(
            values.data(), static_cast<std::size_t>(values.size()));
    });
    return py::make_tuple(stats.minimum, stats.maximum, stats.mean,
                          stats.sum, stats.count);
}

py::array_t<double> finite_values(const DoubleArray& values) {
    require_1d(values, "values");
    std::vector<double> result = run_released([&] {
        return rflux::qflux::finite_values(
            values.data(), static_cast<std::size_t>(values.shape(0)));
    });
    return vector_to_array(result);
}

py::tuple last_finite(const DoubleArray& values) {
    require_1d(values, "values");
    bool found = false;
    const double value = run_released([&] {
        return rflux::qflux::last_finite(
            values.data(), static_cast<std::size_t>(values.shape(0)), &found);
    });
    return py::make_tuple(value, found);
}

py::array_t<std::int64_t> top_k_indices(const DoubleArray& values,
                                       std::size_t k, bool descending) {
    require_1d(values, "values");
    const std::vector<std::size_t> indices = run_released([&] {
        return rflux::qflux::top_k_indices(
            values.data(), static_cast<std::size_t>(values.shape(0)), k,
            descending);
    });
    std::vector<std::int64_t> converted(indices.size());
    for (std::size_t i = 0; i < indices.size(); ++i) {
        converted[i] = static_cast<std::int64_t>(indices[i]);
    }
    return array_from(converted, {
        static_cast<py::ssize_t>(converted.size())});
}

std::size_t argmax_index(const DoubleArray& values) {
    require_1d(values, "values");
    return run_released([&] {
        return rflux::qflux::argmax_index(
            values.data(), static_cast<std::size_t>(values.shape(0)));
    });
}

std::int32_t max_i32(const py::array_t<std::int32_t>& values) {
    if (values.ndim() != 1) {
        throw rflux::InvalidArgument("values must be one-dimensional");
    }
    return run_released([&] {
        return rflux::qflux::max_i32(
            values.data(), static_cast<std::size_t>(values.shape(0)));
    });
}

py::tuple complete_case_covariance(const DoubleArray& returns,
                                   std::size_t ddof) {
    require_2d(returns, "returns");
    std::size_t complete = 0;
    std::vector<double> covariance = run_released([&] {
        return rflux::portfoliolab::complete_case_covariance(
            returns.data(), static_cast<std::size_t>(returns.shape(0)),
            static_cast<std::size_t>(returns.shape(1)), ddof, &complete);
    });
    return py::make_tuple(
        matrix_to_array(covariance, returns.shape(1), returns.shape(1)),
        complete);
}

py::array_t<double> portfolio_series(const DoubleArray& returns,
                                     const DoubleArray& weights) {
    require_2d(returns, "returns");
    require_1d(weights, "weights");
    if (returns.shape(1) != weights.shape(0)) {
        throw rflux::InvalidArgument("weights must have one entry per asset");
    }
    std::vector<double> result = run_released([&] {
        return rflux::portfoliolab::portfolio_series(
            returns.data(), static_cast<std::size_t>(returns.shape(0)),
            static_cast<std::size_t>(returns.shape(1)), weights.data());
    });
    return vector_to_array(result);
}

py::tuple advice_signals(const DoubleArray& prices,
                         const DoubleArray& returns) {
    require_2d(prices, "prices");
    require_2d(returns, "returns");
    if (prices.shape(0) != returns.shape(0) ||
        prices.shape(1) != returns.shape(1)) {
        throw rflux::InvalidArgument(
            "prices and returns must have equal shape");
    }
    const auto n_assets = static_cast<std::size_t>(prices.shape(1));
    const rflux::portfoliolab::AdviceSignals signals = run_released([&] {
        return rflux::portfoliolab::advice_signals(
            prices.data(), returns.data(),
            static_cast<std::size_t>(prices.shape(0)), n_assets);
    });
    std::vector<double> flat(n_assets * 7, 0.0);
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        const auto& row = signals.rows[asset];
        flat[asset * 7 + 0] = row.last_price;
        flat[asset * 7 + 1] = row.momentum_20;
        flat[asset * 7 + 2] = row.momentum_60;
        flat[asset * 7 + 3] = row.trend;
        flat[asset * 7 + 4] = row.rsi_14;
        flat[asset * 7 + 5] = row.daily_sigma;
        flat[asset * 7 + 6] = row.reference_stop;
    }
    return py::make_tuple(
        matrix_to_array(flat, static_cast<py::ssize_t>(n_assets), 7),
        vector_to_array(signals.scores));
}




py::array_t<std::int32_t> row_argmax(const DoubleArray& values) {
    require_2d(values, "values");
    std::vector<std::int32_t> result = run_released([&] {
        return rflux::qflux::row_argmax(
            values.data(), static_cast<std::size_t>(values.shape(0)),
            static_cast<std::size_t>(values.shape(1)));
    });
    return int_vector_to_array(result);
}

double conditional_mean(const DoubleArray& state_probs,
                        const DoubleArray& means, std::size_t feature) {
    require_1d(state_probs, "state_probs");
    require_2d(means, "means");
    if (state_probs.shape(0) != means.shape(0)) {
        throw rflux::InvalidArgument(
            "state_probs must have one entry per state");
    }
    return run_released([&] {
        return rflux::regimelab::conditional_mean(
            state_probs.data(), means.data(),
            static_cast<std::size_t>(means.shape(0)),
            static_cast<std::size_t>(means.shape(1)), feature);
    });
}

py::tuple sticky_regime(const DoubleArray& probabilities,
                        std::size_t min_duration, double entry_threshold,
                        double exit_threshold) {
    require_2d(probabilities, "probabilities");
    const auto n_obs = static_cast<std::size_t>(probabilities.shape(0));
    const auto n_states = static_cast<std::size_t>(probabilities.shape(1));
    const auto result = run_released([&] {
        return rflux::regimelab::sticky_regime(
            probabilities.data(), n_obs, n_states, min_duration,
            entry_threshold, exit_threshold);
    });
    return py::make_tuple(int_vector_to_array(result.labels),
                          int_vector_to_array(result.switches));
}

py::dict align_parameters(const DoubleArray& reference_means,
                          const DoubleArray& reference_variances,
                          const DoubleArray& target_start_prob,
                          const DoubleArray& target_transition,
                          const DoubleArray& target_means,
                          const DoubleArray& target_variances,
                          double variance_weight) {
    require_2d(reference_means, "reference_means");
    require_2d(reference_variances, "reference_variances");
    require_1d(target_start_prob, "target_start_prob");
    require_2d(target_transition, "target_transition");
    require_2d(target_means, "target_means");
    require_2d(target_variances, "target_variances");
    const auto n_states = static_cast<std::size_t>(reference_means.shape(0));
    const auto n_features = static_cast<std::size_t>(reference_means.shape(1));
    if (reference_variances.shape(0) != reference_means.shape(0) ||
        reference_variances.shape(1) != reference_means.shape(1) ||
        target_means.shape(0) != reference_means.shape(0) ||
        target_means.shape(1) != reference_means.shape(1) ||
        target_variances.shape(0) != reference_means.shape(0) ||
        target_variances.shape(1) != reference_means.shape(1) ||
        target_start_prob.shape(0) != reference_means.shape(0) ||
        target_transition.shape(0) != reference_means.shape(0) ||
        target_transition.shape(1) != reference_means.shape(0)) {
        throw rflux::InvalidArgument(
            "state parameter panels must have equal shape");
    }
    const auto result = run_released([&] {
        return rflux::regimelab::align_parameters(
            reference_means.data(), reference_variances.data(),
            target_start_prob.data(), target_transition.data(),
            target_means.data(), target_variances.data(), n_states,
            n_features, variance_weight);
    });
    py::dict output;
    output["order"] = int_vector_to_array(result.order);
    output["start_prob"] = vector_to_array(result.start_prob);
    output["transition"] = matrix_to_array(
        result.transition, static_cast<py::ssize_t>(n_states),
        static_cast<py::ssize_t>(n_states));
    output["means"] = matrix_to_array(
        result.means, static_cast<py::ssize_t>(n_states),
        static_cast<py::ssize_t>(n_features));
    output["variances"] = matrix_to_array(
        result.variances, static_cast<py::ssize_t>(n_states),
        static_cast<py::ssize_t>(n_features));
    output["cost"] = result.cost;
    return output;
}

py::dict regime_attribution(const DoubleArray& returns,
                            const py::array_t<std::int32_t>& labels,
                            double periods_per_year, double risk_free,
                            double confidence) {
    require_1d(returns, "returns");
    if (labels.ndim() != 1 ||
        labels.shape(0) != returns.shape(0)) {
        throw rflux::InvalidArgument(
            "labels must be one-dimensional with one entry per return");
    }
    const auto result = run_released([&] {
        return rflux::regimelab::regime_attribution(
            returns.data(), labels.data(),
            static_cast<std::size_t>(returns.shape(0)), periods_per_year,
            risk_free, confidence);
    });
    const auto count = static_cast<py::ssize_t>(result.state_count);
    std::vector<std::int64_t> periods(result.state_count);
    std::vector<std::int64_t> n_returns(result.state_count);
    std::vector<double> period_share(result.state_count);
    std::vector<double> mean_return(result.state_count);
    std::vector<double> annualized_mean(result.state_count);
    std::vector<double> annualized_volatility(result.state_count);
    std::vector<double> sharpe(result.state_count);
    std::vector<double> var(result.state_count);
    std::vector<double> cvar(result.state_count);
    std::vector<double> hit_rate(result.state_count);
    std::vector<double> sum_return(result.state_count);
    std::vector<double> share_of_total(result.state_count);
    for (std::size_t state = 0; state < result.state_count; ++state) {
        const auto& entry = result.states[state];
        periods[state] = static_cast<std::int64_t>(entry.periods);
        n_returns[state] = static_cast<std::int64_t>(entry.n_returns);
        period_share[state] = entry.period_share;
        mean_return[state] = entry.mean_return;
        annualized_mean[state] = entry.annualized_mean_return;
        annualized_volatility[state] = entry.annualized_volatility;
        sharpe[state] = entry.sharpe;
        var[state] = entry.var;
        cvar[state] = entry.cvar;
        hit_rate[state] = entry.hit_rate;
        sum_return[state] = entry.sum_return;
        share_of_total[state] = entry.share_of_total_sum;
    }
    std::vector<std::int64_t> counts(result.transition_counts.size());
    for (std::size_t i = 0; i < result.transition_counts.size(); ++i) {
        counts[i] = static_cast<std::int64_t>(result.transition_counts[i]);
    }
    py::dict output;
    output["state_count"] = result.state_count;
    output["labelled_observations"] = result.labelled_observations;
    output["periods"] = array_from(periods, {count});
    output["n_returns"] = array_from(n_returns, {count});
    output["period_share"] = vector_to_array(period_share);
    output["mean_return"] = vector_to_array(mean_return);
    output["annualized_mean_return"] = vector_to_array(annualized_mean);
    output["annualized_volatility"] = vector_to_array(annualized_volatility);
    output["sharpe"] = vector_to_array(sharpe);
    output["var"] = vector_to_array(var);
    output["cvar"] = vector_to_array(cvar);
    output["hit_rate"] = vector_to_array(hit_rate);
    output["sum_return"] = vector_to_array(sum_return);
    output["share_of_total_sum"] = vector_to_array(share_of_total);
    output["transition_counts"] = array_from(counts, {count, count});
    output["transition_probabilities"] = matrix_to_array(
        result.transition_probabilities, count, count);
    return output;
}

py::tuple advice_allocation(const DoubleArray& covariance,
                            const DoubleArray& scores,
                            const py::object& current_weights,
                            const DoubleArray& market_returns,
                            double periods_per_year, double max_weight,
                            double target_volatility, double regime_scale) {
    require_2d(covariance, "covariance");
    require_1d(scores, "scores");
    require_1d(market_returns, "market_returns");
    if (covariance.shape(0) != covariance.shape(1) ||
        covariance.shape(0) != scores.shape(0)) {
        throw rflux::InvalidArgument(
            "covariance must be square with one score per asset");
    }
    DoubleArray current;
    const double* current_data = nullptr;
    bool has_current = false;
    if (!current_weights.is_none()) {
        current = py::cast<DoubleArray>(current_weights);
        require_1d(current, "current_weights");
        if (current.shape(0) != covariance.shape(0)) {
            throw rflux::InvalidArgument(
                "current_weights must have one entry per asset");
        }
        current_data = current.data();
        has_current = true;
    }
    const auto allocation = run_released([&] {
        return rflux::portfoliolab::advice_allocation(
            covariance.data(), scores.data(),
            static_cast<std::size_t>(covariance.shape(0)), current_data,
            has_current, market_returns.data(),
            static_cast<std::size_t>(market_returns.shape(0)),
            periods_per_year, max_weight, target_volatility, regime_scale);
    });
    return py::make_tuple(
        vector_to_array(allocation.base_weights),
        vector_to_array(allocation.current_weights),
        vector_to_array(allocation.target_weights),
        vector_to_array(allocation.deltas),
        int_vector_to_array(allocation.actions),
        allocation.market_annualized_volatility,
        allocation.portfolio_annualized_volatility,
        allocation.volatility_scale, allocation.exposure,
        allocation.cash_weight, allocation.current_cash_weight,
        allocation.turnover);
}

py::tuple advice_regime(const DoubleArray& filtered,
                        const DoubleArray& forecast,
                        const DoubleArray& means) {
    require_1d(filtered, "filtered");
    require_1d(forecast, "forecast");
    require_1d(means, "means");
    if (filtered.shape(0) != forecast.shape(0) ||
        filtered.shape(0) != means.shape(0)) {
        throw rflux::InvalidArgument(
            "filtered, forecast and means must have equal lengths");
    }
    const auto regime = run_released([&] {
        return rflux::portfoliolab::advice_regime(
            filtered.data(), forecast.data(), means.data(),
            static_cast<std::size_t>(filtered.shape(0)));
    });
    return py::make_tuple(regime.latest_state, regime.expected_next_return,
                          regime.stress_probability, regime.regime_scale);
}

py::tuple advice_sizing(const DoubleArray& target_weights,
                        const DoubleArray& last_prices,
                        const DoubleArray& reference_stops,
                        const py::object& portfolio_value,
                        double risk_per_trade) {
    require_1d(target_weights, "target_weights");
    require_1d(last_prices, "last_prices");
    require_1d(reference_stops, "reference_stops");
    const auto n_assets = static_cast<std::size_t>(target_weights.shape(0));
    if (last_prices.shape(0) != target_weights.shape(0) ||
        reference_stops.shape(0) != target_weights.shape(0)) {
        throw rflux::InvalidArgument(
            "target_weights, last_prices and reference_stops must match");
    }
    double value = 0.0;
    bool has_value = false;
    if (!portfolio_value.is_none()) {
        value = py::cast<double>(portfolio_value);
        has_value = true;
    }
    const auto sizing = run_released([&] {
        return rflux::portfoliolab::advice_sizing(
            target_weights.data(), last_prices.data(),
            reference_stops.data(), n_assets, value, has_value,
            risk_per_trade);
    });
    std::vector<double> notional(n_assets, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> stop_distance(n_assets, std::numeric_limits<double>::quiet_NaN());
    std::vector<std::int64_t> target_shares(n_assets, -1);
    std::vector<std::int64_t> risk_shares(n_assets, -1);
    std::vector<std::int64_t> recommended(n_assets, -1);
    for (std::size_t asset = 0; asset < n_assets; ++asset) {
        const auto& row = sizing[asset];
        if (row.has_notional) notional[asset] = row.target_notional;
        if (row.has_stop_distance) stop_distance[asset] = row.stop_distance_pct;
        if (row.has_target_shares) target_shares[asset] = row.target_shares;
        if (row.has_risk_limited_shares) {
            risk_shares[asset] = row.risk_limited_shares;
        }
        if (row.has_recommended_shares) {
            recommended[asset] = row.recommended_shares;
        }
    }
    return py::make_tuple(
        vector_to_array(notional), vector_to_array(stop_distance),
        array_from(target_shares, {static_cast<py::ssize_t>(n_assets)}),
        array_from(risk_shares, {static_cast<py::ssize_t>(n_assets)}),
        array_from(recommended, {static_cast<py::ssize_t>(n_assets)}));
}


py::array_t<double> correlation_to_distance(const DoubleArray& correlation) {
    require_2d(correlation, "correlation");
    if (correlation.shape(0) != correlation.shape(1)) {
        throw rflux::InvalidArgument("correlation must be square");
    }
    const auto n = static_cast<std::size_t>(correlation.shape(0));
    std::vector<double> result = run_released([&] {
        return rflux::percolab::correlation_to_distance(
            correlation.data(), n);
    });
    return matrix_to_array(result, correlation.shape(0), correlation.shape(0));
}

py::tuple connected_components(const DoubleArray& adjacency,
                               double threshold) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1)) {
        throw rflux::InvalidArgument("adjacency must be square");
    }
    const auto n = static_cast<std::size_t>(adjacency.shape(0));
    const auto result = run_released([&] {
        return rflux::percolab::connected_components(
            adjacency.data(), n, threshold);
    });
    const std::vector<std::size_t> sizes = rflux::percolab::component_sizes(
        result.labels.data(), n, result.n_components);
    return py::make_tuple(int_vector_to_array(result.labels),
                          result.n_components, result.largest_size,
                          array_from(sizes, {static_cast<py::ssize_t>(sizes.size())}));
}

py::array_t<double> network_strength(const DoubleArray& adjacency) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1)) {
        throw rflux::InvalidArgument("adjacency must be square");
    }
    std::vector<double> result = run_released([&] {
        return rflux::percolab::strength(
            adjacency.data(), static_cast<std::size_t>(adjacency.shape(0)));
    });
    return vector_to_array(result);
}

py::array_t<double> centrality(const DoubleArray& adjacency,
                               std::size_t kind) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1) || kind > 1) {
        throw rflux::InvalidArgument(
            "adjacency must be square and kind must be 0 or 1");
    }
    std::vector<double> result = run_released([&] {
        if (kind == 0) {
            return rflux::percolab::strength(
                adjacency.data(),
                static_cast<std::size_t>(adjacency.shape(0)));
        }
        return rflux::percolab::eigenvector_centrality(
            adjacency.data(),
            static_cast<std::size_t>(adjacency.shape(0)));
    });
    return vector_to_array(result);
}

py::array_t<double> clustering(const DoubleArray& adjacency) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1)) {
        throw rflux::InvalidArgument("adjacency must be square");
    }
    std::vector<double> result = run_released([&] {
        return rflux::percolab::clustering_coefficients(
            adjacency.data(), static_cast<std::size_t>(adjacency.shape(0)));
    });
    return vector_to_array(result);
}

double transitivity(const DoubleArray& adjacency) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1)) {
        throw rflux::InvalidArgument("adjacency must be square");
    }
    return run_released([&] {
        return rflux::percolab::transitivity(
            adjacency.data(), static_cast<std::size_t>(adjacency.shape(0)));
    });
}

py::tuple path_statistics(const DoubleArray& adjacency) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1)) {
        throw rflux::InvalidArgument("adjacency must be square");
    }
    const auto result = run_released([&] {
        return rflux::percolab::path_statistics(
            adjacency.data(), static_cast<std::size_t>(adjacency.shape(0)));
    });
    return py::make_tuple(result.efficiency, result.average_path_length,
                          result.diameter, result.n_reachable_pairs,
                          result.connected);
}

py::array_t<std::int32_t> core_numbers(const DoubleArray& adjacency) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1)) {
        throw rflux::InvalidArgument("adjacency must be square");
    }
    std::vector<std::int32_t> result = run_released([&] {
        return rflux::percolab::core_numbers(
            adjacency.data(), static_cast<std::size_t>(adjacency.shape(0)));
    });
    return int_vector_to_array(result);
}

double assortativity(const DoubleArray& adjacency) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1)) {
        throw rflux::InvalidArgument("adjacency must be square");
    }
    return run_released([&] {
        return rflux::percolab::degree_assortativity(
            adjacency.data(), static_cast<std::size_t>(adjacency.shape(0)));
    });
}

double modularity(const DoubleArray& adjacency,
                  const py::array_t<std::int32_t>& labels) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1) ||
        labels.ndim() != 1 || labels.shape(0) != adjacency.shape(0)) {
        throw rflux::InvalidArgument(
            "adjacency must be square and labels must have one entry per node");
    }
    const auto n = static_cast<std::size_t>(adjacency.shape(0));
    std::map<std::int32_t, std::int32_t> mapping;
    std::vector<std::int32_t> contiguous(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const std::int32_t value = labels.data()[i];
        if (value < 0) {
            throw rflux::InvalidArgument("labels must be non-negative");
        }
        auto [iterator, inserted] =
            mapping.emplace(value, static_cast<std::int32_t>(mapping.size()));
        contiguous[i] = iterator->second;
    }
    return run_released([&] {
        return rflux::percolab::modularity(
            adjacency.data(), n, contiguous.data(), mapping.size());
    });
}

py::array_t<double> pagerank(const DoubleArray& adjacency, double damping,
                             std::size_t max_iter, double tol) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1)) {
        throw rflux::InvalidArgument("adjacency must be square");
    }
    std::vector<double> result = run_released([&] {
        return rflux::percolab::pagerank(
            adjacency.data(), static_cast<std::size_t>(adjacency.shape(0)),
            damping, max_iter, tol);
    });
    return vector_to_array(result);
}

py::array_t<double> betweenness(const DoubleArray& adjacency,
                                bool normalized) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1)) {
        throw rflux::InvalidArgument("adjacency must be square");
    }
    std::vector<double> result = run_released([&] {
        return rflux::percolab::betweenness_centrality(
            adjacency.data(), static_cast<std::size_t>(adjacency.shape(0)),
            normalized);
    });
    return vector_to_array(result);
}

py::array_t<std::uint32_t> bridges(const DoubleArray& adjacency) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1)) {
        throw rflux::InvalidArgument("adjacency must be square");
    }
    const auto n = static_cast<std::size_t>(adjacency.shape(0));
    std::vector<rflux::percolab::BridgeEdge> edges = run_released([&] {
        return rflux::percolab::bridges(adjacency.data(), n);
    });
    std::vector<std::uint32_t> flat(edges.size() * 2);
    for (std::size_t i = 0; i < edges.size(); ++i) {
        flat[i * 2] = edges[i].u;
        flat[i * 2 + 1] = edges[i].v;
    }
    return array_from(flat, {static_cast<py::ssize_t>(edges.size()), 2});
}

py::array_t<std::int32_t> articulation_points(const DoubleArray& adjacency) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1)) {
        throw rflux::InvalidArgument("adjacency must be square");
    }
    std::vector<std::int32_t> result = run_released([&] {
        return rflux::percolab::articulation_points(
            adjacency.data(), static_cast<std::size_t>(adjacency.shape(0)));
    });
    return int_vector_to_array(result);
}

py::array_t<std::int32_t> truss_numbers(const DoubleArray& adjacency) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1)) {
        throw rflux::InvalidArgument("adjacency must be square");
    }
    std::vector<std::int32_t> result = run_released([&] {
        return rflux::percolab::truss_numbers(
            adjacency.data(), static_cast<std::size_t>(adjacency.shape(0)));
    });
    return array_from(result,
                      {adjacency.shape(0), adjacency.shape(0)});
}

py::tuple minimum_spanning_tree(const DoubleArray& distance) {
    require_2d(distance, "distance");
    if (distance.shape(0) != distance.shape(1)) {
        throw rflux::InvalidArgument("distance must be square");
    }
    const auto edges = run_released([&] {
        return rflux::percolab::minimum_spanning_tree(
            distance.data(), static_cast<std::size_t>(distance.shape(0)));
    });
    std::vector<std::uint32_t> u(edges.size());
    std::vector<std::uint32_t> v(edges.size());
    std::vector<double> weight(edges.size());
    for (std::size_t i = 0; i < edges.size(); ++i) {
        u[i] = edges[i].u;
        v[i] = edges[i].v;
        weight[i] = edges[i].weight;
    }
    const auto count = static_cast<py::ssize_t>(edges.size());
    return py::make_tuple(array_from(u, {count}), array_from(v, {count}),
                          vector_to_array(weight));
}

py::tuple mst_clusters(const DoubleArray& distance, double threshold) {
    require_2d(distance, "distance");
    if (distance.shape(0) != distance.shape(1)) {
        throw rflux::InvalidArgument("distance must be square");
    }
    const auto n = static_cast<std::size_t>(distance.shape(0));
    const auto result = run_released([&] {
        return rflux::percolab::mst_clusters(distance.data(), n, threshold);
    });
    const std::vector<std::size_t> sizes = rflux::percolab::component_sizes(
        result.labels.data(), n, result.n_components);
    return py::make_tuple(int_vector_to_array(result.labels),
                          result.n_components, result.largest_size,
                          array_from(sizes, {static_cast<py::ssize_t>(sizes.size())}));
}

py::tuple percolation_scan(const DoubleArray& adjacency,
                           const DoubleArray& probabilities, bool site,
                           const py::object& edge_weights,
                           const py::object& node_weights,
                           std::size_t repetitions, std::uint64_t seed) {
    require_2d(adjacency, "adjacency");
    require_1d(probabilities, "probabilities");
    if (adjacency.shape(0) != adjacency.shape(1)) {
        throw rflux::InvalidArgument("adjacency must be square");
    }
    rflux::percolab::OccupationWeights weights;
    DoubleArray edge;
    DoubleArray node;
    if (!edge_weights.is_none()) {
        edge = py::cast<DoubleArray>(edge_weights);
        require_2d(edge, "edge_weights");
        if (edge.shape(0) != adjacency.shape(0) ||
            edge.shape(1) != adjacency.shape(1)) {
            throw rflux::InvalidArgument(
                "edge_weights must have the same shape as adjacency");
        }
        weights.edge = edge.data();
    }
    if (!node_weights.is_none()) {
        node = py::cast<DoubleArray>(node_weights);
        require_1d(node, "node_weights");
        if (node.shape(0) != adjacency.shape(0)) {
            throw rflux::InvalidArgument(
                "node_weights must have one entry per node");
        }
        weights.node = node.data();
    }
    const auto points = run_released([&] {
        return rflux::percolab::percolation_scan(
            adjacency.data(), static_cast<std::size_t>(adjacency.shape(0)),
            site ? rflux::percolab::PercolationKind::site
                 : rflux::percolab::PercolationKind::bond,
            weights, probabilities.data(),
            static_cast<std::size_t>(probabilities.shape(0)), repetitions,
            seed);
    });
    std::vector<double> mean(points.size());
    std::vector<double> stddev(points.size());
    std::vector<double> susceptibility(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        mean[i] = points[i].giant_mean;
        stddev[i] = points[i].giant_std;
        susceptibility[i] = points[i].susceptibility;
    }
    return py::make_tuple(vector_to_array(mean), vector_to_array(stddev),
                          vector_to_array(susceptibility));
}

py::array_t<double> cluster_size_distribution(
    const DoubleArray& adjacency, double probability, bool site,
    const py::object& edge_weights, const py::object& node_weights,
    std::size_t repetitions, std::uint64_t seed) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1)) {
        throw rflux::InvalidArgument("adjacency must be square");
    }
    rflux::percolab::OccupationWeights weights;
    DoubleArray edge;
    DoubleArray node;
    if (!edge_weights.is_none()) {
        edge = py::cast<DoubleArray>(edge_weights);
        require_2d(edge, "edge_weights");
        if (edge.shape(0) != adjacency.shape(0) ||
            edge.shape(1) != adjacency.shape(1)) {
            throw rflux::InvalidArgument(
                "edge_weights must have the same shape as adjacency");
        }
        weights.edge = edge.data();
    }
    if (!node_weights.is_none()) {
        node = py::cast<DoubleArray>(node_weights);
        require_1d(node, "node_weights");
        if (node.shape(0) != adjacency.shape(0)) {
            throw rflux::InvalidArgument(
                "node_weights must have one entry per node");
        }
        weights.node = node.data();
    }
    const auto result = run_released([&] {
        return rflux::percolab::cluster_size_distribution(
            adjacency.data(), static_cast<std::size_t>(adjacency.shape(0)),
            site ? rflux::percolab::PercolationKind::site
                 : rflux::percolab::PercolationKind::bond,
            weights, probability, repetitions, seed);
    });
    return vector_to_array(result);
}

py::array_t<std::uint32_t> removal_order(const DoubleArray& adjacency,
                                         std::size_t strategy) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1) || strategy > 4) {
        throw rflux::InvalidArgument(
            "adjacency must be square and strategy must be in 0..4");
    }
    const auto result = run_released([&] {
        return rflux::percolab::removal_order(
            adjacency.data(), static_cast<std::size_t>(adjacency.shape(0)),
            static_cast<rflux::percolab::RemovalStrategy>(strategy));
    });
    return array_from(result, {static_cast<py::ssize_t>(result.size())});
}

py::tuple targeted_removal(const DoubleArray& adjacency,
                           const py::array_t<std::uint32_t>& order,
                           std::size_t checkpoints) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1) ||
        order.ndim() != 1 || order.shape(0) != adjacency.shape(0)) {
        throw rflux::InvalidArgument(
            "adjacency must be square and order must have one entry per node");
    }
    const auto n = static_cast<std::size_t>(adjacency.shape(0));
    const auto points = run_released([&] {
        return rflux::percolab::targeted_removal(
            adjacency.data(), n, order.data(), checkpoints);
    });
    const auto count = static_cast<py::ssize_t>(points.size());
    std::vector<std::int64_t> removed(points.size());
    std::vector<double> giant(points.size());
    std::vector<double> susceptibility(points.size());
    std::vector<std::int64_t> n_components(points.size());
    for (std::size_t i = 0; i < points.size(); ++i) {
        removed[i] = static_cast<std::int64_t>(points[i].removed);
        giant[i] = points[i].giant_fraction;
        susceptibility[i] = points[i].finite_susceptibility;
        n_components[i] = static_cast<std::int64_t>(points[i].n_components);
    }
    return py::make_tuple(array_from(removed, {count}),
                          vector_to_array(giant),
                          vector_to_array(susceptibility),
                          array_from(n_components, {count}));
}

py::tuple core_percolation(const DoubleArray& adjacency) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1)) {
        throw rflux::InvalidArgument("adjacency must be square");
    }
    const auto result = run_released([&] {
        return rflux::percolab::core_percolation(
            adjacency.data(), static_cast<std::size_t>(adjacency.shape(0)));
    });
    const auto count = static_cast<py::ssize_t>(result.k.size());
    return py::make_tuple(array_from(result.k, {count}),
                          vector_to_array(result.fraction),
                          array_from(result.n_nodes, {count}));
}

py::tuple risk_diffusion(const DoubleArray& adjacency,
                         const py::object& seeds, double beta, double gamma,
                         std::size_t steps, std::size_t repetitions,
                         std::uint64_t seed) {
    require_2d(adjacency, "adjacency");
    if (adjacency.shape(0) != adjacency.shape(1)) {
        throw rflux::InvalidArgument("adjacency must be square");
    }
    py::array_t<std::uint32_t> seed_array;
    const std::uint32_t* seed_data = nullptr;
    std::size_t seed_count = 0;
    if (!seeds.is_none()) {
        seed_array = py::cast<py::array_t<std::uint32_t>>(seeds);
        if (seed_array.ndim() != 1) {
            throw rflux::InvalidArgument("seeds must be one-dimensional");
        }
        seed_data = seed_array.data();
        seed_count = static_cast<std::size_t>(seed_array.shape(0));
    }
    const auto result = run_released([&] {
        return rflux::percolab::risk_diffusion(
            adjacency.data(), static_cast<std::size_t>(adjacency.shape(0)),
            seed_data, seed_count, beta, gamma, steps, repetitions, seed);
    });
    return py::make_tuple(vector_to_array(result.mean_affected),
                          vector_to_array(result.infection_frequency));
}

py::tuple correlation_adjacency(const DoubleArray& correlation,
                                double threshold, bool weighted) {
    require_2d(correlation, "correlation");
    if (correlation.shape(0) != correlation.shape(1)) {
        throw rflux::InvalidArgument("correlation must be square");
    }
    const auto result = run_released([&] {
        return rflux::percolab::correlation_adjacency(
            correlation.data(),
            static_cast<std::size_t>(correlation.shape(0)), threshold,
            weighted);
    });
    return py::make_tuple(
        matrix_to_array(result.adjacency, correlation.shape(0),
                        correlation.shape(0)),
        result.n_edges, result.density);
}

py::tuple percolation_summary(const DoubleArray& probabilities,
                              const DoubleArray& giant_mean,
                              const DoubleArray& susceptibility) {
    require_1d(probabilities, "probabilities");
    require_1d(giant_mean, "giant_mean");
    require_1d(susceptibility, "susceptibility");
    if (probabilities.shape(0) != giant_mean.shape(0) ||
        probabilities.shape(0) != susceptibility.shape(0)) {
        throw rflux::InvalidArgument("scan arrays must have equal lengths");
    }
    const auto result = run_released([&] {
        return rflux::percolab::percolation_summary(
            probabilities.data(), giant_mean.data(), susceptibility.data(),
            static_cast<std::size_t>(probabilities.shape(0)));
    });
    return py::make_tuple(result.critical_probability, result.p_half_giant,
                          result.max_susceptibility, result.giant_area);
}

py::tuple volatility_regime(const DoubleArray& returns,
                            std::size_t vol_window, std::size_t lookback,
                            double low_q, double high_q) {
    require_1d(returns, "returns");
    const auto result = run_released([&] {
        return rflux::regimelab::volatility_regime(
            returns.data(), static_cast<std::size_t>(returns.shape(0)),
            vol_window, lookback, low_q, high_q);
    });
    return py::make_tuple(vector_to_array(result.volatility),
                          int_vector_to_array(result.labels));
}

py::tuple trend_regime(const DoubleArray& prices, std::size_t fast_window,
                       std::size_t slow_window, double threshold) {
    require_1d(prices, "prices");
    const auto result = run_released([&] {
        return rflux::regimelab::trend_regime(
            prices.data(), static_cast<std::size_t>(prices.shape(0)),
            fast_window, slow_window, threshold);
    });
    return py::make_tuple(vector_to_array(result.fast_ma),
                          vector_to_array(result.slow_ma),
                          int_vector_to_array(result.labels));
}

py::array_t<std::int32_t> composite_regime(
    const py::array_t<std::int32_t>& volatility_labels,
    const py::array_t<std::int32_t>& trend_labels) {
    if (volatility_labels.ndim() != 1 || trend_labels.ndim() != 1 ||
        volatility_labels.shape(0) != trend_labels.shape(0)) {
        throw rflux::InvalidArgument(
            "labels must be one-dimensional with equal lengths");
    }
    return int_vector_to_array(run_released([&] {
        return rflux::regimelab::composite_regime(
            volatility_labels.data(), trend_labels.data(),
            static_cast<std::size_t>(volatility_labels.shape(0)));
    }));
}

}  // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "rflux native C++ bridge (pybind11, zero-copy, GIL-released)";
    module.attr("__version__") = RFLUX_VERSION_STRING;
    module.attr("ABI_VERSION") = py::int_(RFLUX_ABI_VERSION);
    module.attr("STABLE_ABI") = static_cast<bool>(RFLUX_NATIVE_STABLE_ABI != 0);
    module.attr("GIL_RELEASED") = true;

    py::object native_error =
        py::reinterpret_steal<py::object>(PyErr_NewException(
            "rflux._native.NativeError", PyExc_RuntimeError, nullptr));
    g_native_error = native_error.ptr();
    module.add_object("NativeError", native_error);
    register_error_translator();

    module.def(
        "series",
        [](const std::string& name, const DoubleArray& values,
           const py::args& args) { return series(name, values, args); },
        py::arg("name"), py::arg("values"));
    module.def(
        "series_into",
        [](const std::string& name, const DoubleArray& values, py::array out,
           const py::args& args) {
            return series_into(name, values, std::move(out), args);
        },
        py::arg("name"), py::arg("values"), py::arg("out"));
    module.def(
        "pair_series",
        [](const std::string& name, const DoubleArray& x, const DoubleArray& y,
           const py::args& args) { return pair_series(name, x, y, args); },
        py::arg("name"), py::arg("x"), py::arg("y"));
    module.def("atr", &atr_series, py::arg("high"), py::arg("low"),
               py::arg("close"), py::arg("period") = 14);
    module.def("macd", &macd_result, py::arg("prices"), py::arg("fast") = 12,
               py::arg("slow") = 26, py::arg("signal") = 9);
    module.def("bollinger", &bollinger_result, py::arg("prices"),
               py::arg("window") = 20, py::arg("n_std") = 2.0);
    module.def("describe", &describe_values, py::arg("values"));
    module.def("quantile", &quantile_value, py::arg("values"), py::arg("q"));
    module.def("pearson", &pearson_value, py::arg("x"), py::arg("y"));
    module.def("correlation_matrix", &correlation_matrix_values,
               py::arg("panel"));
    module.def("sharpe", &sharpe_value, py::arg("returns"),
               py::arg("periods_per_year") = 252.0,
               py::arg("risk_free") = 0.0);
    module.def("sortino", &sortino_value, py::arg("returns"),
               py::arg("periods_per_year") = 252.0,
               py::arg("risk_free") = 0.0);
    module.def("var_cvar", &var_cvar_value, py::arg("returns"),
               py::arg("confidence") = 0.95);
    module.def("equity_curve", &equity_curve, py::arg("returns"),
               py::arg("initial") = 1.0);
    module.def("performance_summary", &performance_summary, py::arg("returns"),
               py::arg("periods_per_year") = 252.0,
               py::arg("risk_free") = 0.0, py::arg("confidence") = 0.95,
               py::arg("periods") = 0);
    module.def("neutralize", &neutralize_values, py::arg("values"),
               py::arg("groups"), py::arg("group_count"));
    module.def("equal_weight_market_returns", &equal_weight_market_returns,
               py::arg("returns"));
    module.def("finite_stats", &finite_stats, py::arg("values"));
    module.def("finite_values", &finite_values, py::arg("values"));
    module.def("last_finite", &last_finite, py::arg("values"));
    module.def("top_k_indices", &top_k_indices, py::arg("values"),
               py::arg("k"), py::arg("descending") = true);
    module.def("argmax_index", &argmax_index, py::arg("values"));
    module.def("max_i32", &max_i32, py::arg("values"));
    module.def("complete_case_covariance", &complete_case_covariance,
               py::arg("returns"), py::arg("ddof") = 1);
    module.def("portfolio_series", &portfolio_series, py::arg("returns"),
               py::arg("weights"));
    module.def("row_argmax", &row_argmax, py::arg("values"));
    module.def("conditional_mean", &conditional_mean,
               py::arg("state_probs"), py::arg("means"),
               py::arg("feature") = 0);
    module.def("sticky_regime", &sticky_regime,
               py::arg("probabilities"), py::arg("min_duration") = 5,
               py::arg("entry_threshold") = 0.6,
               py::arg("exit_threshold") = 0.4);
    module.def("align_parameters", &align_parameters,
               py::arg("reference_means"), py::arg("reference_variances"),
               py::arg("target_start_prob"), py::arg("target_transition"),
               py::arg("target_means"), py::arg("target_variances"),
               py::arg("variance_weight") = 1.0);
    module.def("correlation_to_distance", &correlation_to_distance,
               py::arg("correlation"));
    module.def("connected_components", &connected_components,
               py::arg("adjacency"), py::arg("threshold"));
    module.def("network_strength", &network_strength, py::arg("adjacency"));
    module.def("centrality", &centrality, py::arg("adjacency"),
               py::arg("kind"));
    module.def("clustering", &clustering, py::arg("adjacency"));
    module.def("transitivity", &transitivity, py::arg("adjacency"));
    module.def("path_statistics", &path_statistics, py::arg("adjacency"));
    module.def("core_numbers", &core_numbers, py::arg("adjacency"));
    module.def("assortativity", &assortativity, py::arg("adjacency"));
    module.def("modularity", &modularity, py::arg("adjacency"),
               py::arg("labels"));
    module.def("pagerank", &pagerank, py::arg("adjacency"),
               py::arg("damping") = 0.85, py::arg("max_iter") = 100,
               py::arg("tol") = 1e-8);
    module.def("betweenness", &betweenness, py::arg("adjacency"),
               py::arg("normalized") = true);
    module.def("bridges", &bridges, py::arg("adjacency"));
    module.def("articulation_points", &articulation_points,
               py::arg("adjacency"));
    module.def("truss_numbers", &truss_numbers, py::arg("adjacency"));
    module.def("minimum_spanning_tree", &minimum_spanning_tree,
               py::arg("distance"));
    module.def("mst_clusters", &mst_clusters, py::arg("distance"),
               py::arg("threshold"));
    module.def("percolation_scan", &percolation_scan, py::arg("adjacency"),
               py::arg("probabilities"), py::arg("site") = false,
               py::arg("edge_weights") = py::none(),
               py::arg("node_weights") = py::none(),
               py::arg("repetitions") = 100, py::arg("seed") = 42);
    module.def("cluster_size_distribution", &cluster_size_distribution,
               py::arg("adjacency"), py::arg("probability"),
               py::arg("site") = false,
               py::arg("edge_weights") = py::none(),
               py::arg("node_weights") = py::none(),
               py::arg("repetitions") = 100, py::arg("seed") = 42);
    module.def("removal_order", &removal_order, py::arg("adjacency"),
               py::arg("strategy"));
    module.def("targeted_removal", &targeted_removal,
               py::arg("adjacency"), py::arg("order"),
               py::arg("checkpoints") = 10);
    module.def("core_percolation", &core_percolation, py::arg("adjacency"));
    module.def("risk_diffusion", &risk_diffusion, py::arg("adjacency"),
               py::arg("seeds") = py::none(), py::arg("beta") = 0.3,
               py::arg("gamma") = 0.1, py::arg("steps") = 30,
               py::arg("repetitions") = 100, py::arg("seed") = 42);
    module.def("correlation_adjacency", &correlation_adjacency,
               py::arg("correlation"), py::arg("threshold"),
               py::arg("weighted") = true);
    module.def("percolation_summary", &percolation_summary,
               py::arg("probabilities"), py::arg("giant_mean"),
               py::arg("susceptibility"));
    module.def("volatility_regime", &volatility_regime, py::arg("returns"),
               py::arg("vol_window") = 20, py::arg("lookback") = 252,
               py::arg("low_q") = 0.25, py::arg("high_q") = 0.75);
    module.def("trend_regime", &trend_regime, py::arg("prices"),
               py::arg("fast_window") = 20, py::arg("slow_window") = 60,
               py::arg("threshold") = 0.0);
    module.def("composite_regime", &composite_regime,
               py::arg("volatility_labels"), py::arg("trend_labels"));
    module.def("regime_attribution", &regime_attribution,
               py::arg("returns"), py::arg("labels"),
               py::arg("periods_per_year") = 252.0,
               py::arg("risk_free") = 0.0,
               py::arg("confidence") = 0.95);
    module.def("advice_signals", &advice_signals, py::arg("prices"),
               py::arg("returns"));
    module.def("advice_allocation", &advice_allocation,
               py::arg("covariance"), py::arg("scores"),
               py::arg("current_weights") = py::none(),
               py::arg("market_returns"), py::arg("periods_per_year"),
               py::arg("max_weight"), py::arg("target_volatility"),
               py::arg("regime_scale"));
    module.def("advice_regime", &advice_regime, py::arg("filtered"),
               py::arg("forecast"), py::arg("means"));
    module.def("advice_sizing", &advice_sizing, py::arg("target_weights"),
               py::arg("last_prices"), py::arg("reference_stops"),
               py::arg("portfolio_value") = py::none(),
               py::arg("risk_per_trade") = 0.005);
    module.def("factor_momentum", &factor_momentum, py::arg("close"),
               py::arg("window"));
    module.def("factor_volume_reversal", &factor_volume_reversal,
               py::arg("close"), py::arg("volume"), py::arg("window"));
    module.def("factor_vwap_deviation", &factor_vwap_deviation,
               py::arg("close"), py::arg("volume"), py::arg("window"));
    module.def("factor_ic", &factor_ic, py::arg("factor"),
               py::arg("forward_returns"), py::arg("spearman") = false,
               py::arg("min_pairs") = 2);
    module.def("factor_turnover", &factor_turnover, py::arg("weights"));
    module.def("factor_amihud", &factor_amihud, py::arg("close"),
               py::arg("volume"), py::arg("window"), py::arg("scale") = 1e6);
    module.def("factor_forward_returns", &factor_forward_returns,
               py::arg("prices"), py::arg("periods"), py::arg("log") = false);
    module.def("factor_quantile_returns", &factor_quantile_returns,
               py::arg("factor"), py::arg("forward_returns"),
               py::arg("n_quantiles"));
    module.def("factor_long_short", &factor_long_short, py::arg("factor"),
               py::arg("forward_returns"), py::arg("n_quantiles"),
               py::arg("cost_bps") = 0.0);
    module.def("ic_summary", &ic_summary, py::arg("ic"));
    module.def("regime_match_states", &regime_match_states,
               py::arg("reference_means"), py::arg("reference_variances"),
               py::arg("target_means"), py::arg("target_variances"),
               py::arg("variance_weight") = 1.0);
    module.def("tail_dependence", &tail_dependence, py::arg("returns"),
               py::arg("q") = 0.05, py::arg("upper") = false);
    module.def("hrp_weights", &hrp_weights, py::arg("covariance"));
    module.def("hrp_order", &hrp_order, py::arg("covariance"));
    module.def("portfolio_risk", &portfolio_risk, py::arg("returns"),
               py::arg("weights"), py::arg("ddof") = 1,
               py::arg("periods_per_year") = 252.0);
    module.def("risk_parity_weights", &risk_parity_weights,
               py::arg("covariance"));
    module.def("nco_weights", &nco_weights, py::arg("covariance"),
               py::arg("distance_threshold") = 0.5);
    module.def("rebalance_targets", &rebalance_targets, py::arg("current"),
               py::arg("target"), py::arg("threshold") = 0.05);
    module.def("almgren_chriss", &almgren_chriss, py::arg("quantity"),
               py::arg("slices"), py::arg("horizon"),
               py::arg("temporary_impact"), py::arg("permanent_impact"),
               py::arg("volatility"), py::arg("risk_aversion"));
    module.def("huber_regression", &huber_regression, py::arg("features"),
               py::arg("target"), py::arg("epsilon") = 1.35,
               py::arg("max_iter") = 100, py::arg("tol") = 1e-8,
               py::arg("fit_intercept") = true);
    module.def("linear_predict", &linear_predict,
               py::arg("features"), py::arg("coefficients"),
               py::arg("intercept") = 0.0);
    module.def("cvar_loss", &cvar_loss, py::arg("returns"),
               py::arg("alpha") = 0.95);
    module.def("graph_smooth", &graph_smooth, py::arg("signal"),
               py::arg("adjacency"), py::arg("alpha") = 0.5,
               py::arg("steps") = 3);
    module.def("order_book_metrics", &order_book_metrics, py::arg("bid_price"),
               py::arg("bid_size"), py::arg("ask_price"),
               py::arg("ask_size"));

    py::class_<NativeHMM>(module, "GaussianHMM")
        .def(py::init<std::size_t, std::size_t>(), py::arg("n_states"),
             py::arg("n_features") = 1)
        .def("fit", &NativeHMM::fit, py::arg("observations"),
             py::arg("max_iter") = 200, py::arg("n_init") = 3,
             py::arg("tol") = 1e-6, py::arg("variance_floor") = 1e-6,
             py::arg("seed") = 42, py::arg("sort_states") = true)
        .def_property_readonly("fit_info", &NativeHMM::fit_info)
        .def("score", &NativeHMM::score, py::arg("observations"))
        .def_property_readonly("parameters", &NativeHMM::parameters)
        .def("set_parameters", &NativeHMM::set_parameters, py::arg("start_prob"),
             py::arg("transition"), py::arg("means"), py::arg("variances"))
        .def("viterbi", &NativeHMM::viterbi, py::arg("observations"))
        .def("filtered", &NativeHMM::filtered, py::arg("observations"))
        .def("smoothed", &NativeHMM::smoothed, py::arg("observations"))
        .def("forecast", &NativeHMM::forecast, py::arg("state_probs"),
             py::arg("horizon") = 1)
        .def("filter_step", &NativeHMM::filter_step, py::arg("state_probs"),
             py::arg("observation"))
        .def("stationary_distribution", &NativeHMM::stationary_distribution)
        .def("expected_durations", &NativeHMM::expected_durations)
        .def_property_readonly("n_states", &NativeHMM::n_states)
        .def_property_readonly("n_features", &NativeHMM::n_features)
        .def_property_readonly("ready", &NativeHMM::ready);
}
