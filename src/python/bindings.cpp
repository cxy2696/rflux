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
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "internal/common.hpp"
#include "rflux/execution/almgren_chriss.hpp"
#include "rflux/factorlab/factors.hpp"
#include "rflux/mllab/huber.hpp"
#include "rflux/portfoliolab/hrp.hpp"
#include "rflux/qflux/indicators.hpp"
#include "rflux/qflux/ranking.hpp"
#include "rflux/qflux/stats.hpp"
#include "rflux/qflux/window.hpp"
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

py::array_t<double> factor_amihud(const DoubleArray& close,
                                  const DoubleArray& volume,
                                  std::size_t window, double scale) {
    return window_factor(close, &volume, [window, scale] {
        return rflux::factorlab::AmihudFactor(window, scale);
    });
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

}  // namespace

PYBIND11_MODULE(_native, module) {
    module.doc() = "rflux native C++ bridge (pybind11, zero-copy, GIL-released)";
    module.attr("__version__") = RFLUX_VERSION_STRING;
    module.attr("ABI_VERSION") = 1;
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
    module.def("neutralize", &neutralize_values, py::arg("values"),
               py::arg("groups"), py::arg("group_count"));
    module.def("factor_momentum", &factor_momentum, py::arg("close"),
               py::arg("window"));
    module.def("factor_volume_reversal", &factor_volume_reversal,
               py::arg("close"), py::arg("volume"), py::arg("window"));
    module.def("factor_vwap_deviation", &factor_vwap_deviation,
               py::arg("close"), py::arg("volume"), py::arg("window"));
    module.def("factor_amihud", &factor_amihud, py::arg("close"),
               py::arg("volume"), py::arg("window"), py::arg("scale") = 1e6);
    module.def("tail_dependence", &tail_dependence, py::arg("returns"),
               py::arg("q") = 0.05, py::arg("upper") = false);
    module.def("hrp_weights", &hrp_weights, py::arg("covariance"));
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
