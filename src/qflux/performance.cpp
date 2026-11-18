#include "rflux/qflux/performance.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include "internal/common.hpp"
#include "rflux/qflux/indicators.hpp"
#include "rflux/qflux/stats.hpp"

namespace rflux::qflux {

std::vector<double> equity_curve(const double* returns, std::size_t n,
                                 double initial) {
    rflux::detail::require_pointer(returns, "returns must not be NULL");
    rflux::detail::require(std::isfinite(initial), "initial must be finite");
    std::vector<double> result(n, 0.0);
    double value = initial;
    for (std::size_t i = 0; i < n; ++i) {
        const double observation = returns[i];
        if (std::isfinite(observation)) {
            rflux::detail::require(observation > -1.0,
                                   "returns must be greater than -1");
            value *= 1.0 + observation;
        }
        result[i] = value;
    }
    return result;
}

PerformanceSummary performance_summary(const double* returns, std::size_t n,
                                       double periods_per_year,
                                       double risk_free, double confidence,
                                       std::size_t periods) {
    rflux::detail::require_pointer(returns, "returns must not be NULL");
    rflux::detail::require(std::isfinite(periods_per_year) &&
                               periods_per_year > 0.0,
                           "periods_per_year must be positive");
    rflux::detail::require(std::isfinite(risk_free),
                           "risk_free must be finite");
    rflux::detail::require(std::isfinite(confidence) && confidence > 0.0 &&
                               confidence < 1.0,
                           "confidence must be in (0, 1)");
    std::vector<double> finite;
    finite.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double value = returns[i];
        if (std::isfinite(value)) finite.push_back(value);
    }
    rflux::detail::require(finite.size() >= 2,
                           "at least two finite returns are required");
    const std::size_t elapsed = periods == 0 ? n : periods;
    rflux::detail::require(elapsed > 0, "periods must be positive");

    PerformanceSummary result;
    result.n_periods = finite.size();
    result.elapsed_periods = elapsed;
    long double product = 1.0L;
    double minimum = finite.front();
    for (double value : finite) {
        product *= 1.0L + static_cast<long double>(value);
        minimum = std::min(minimum, value);
    }
    result.total_return = static_cast<double>(product - 1.0L);
    if (minimum > -1.0 && product > 0.0L &&
        std::isfinite(result.total_return)) {
        result.annualized_return = std::pow(
            1.0 + result.total_return,
            periods_per_year / static_cast<double>(elapsed)) - 1.0;
    } else {
        result.annualized_return =
            std::numeric_limits<double>::quiet_NaN();
    }

    long double sum = 0.0L;
    for (double value : finite) sum += value;
    const long double mean = sum / static_cast<long double>(finite.size());
    long double variance = 0.0L;
    for (double value : finite) {
        const long double diff =
            static_cast<long double>(value) - mean;
        variance += diff * diff;
    }
    variance /= static_cast<long double>(finite.size() - 1);
    result.annualized_volatility =
        std::sqrt(static_cast<double>(variance)) * std::sqrt(periods_per_year);

    std::vector<double> equity(finite.size() + 1, 1.0);
    for (std::size_t i = 0; i < finite.size(); ++i) {
        equity[i + 1] = equity[i] * (1.0 + finite[i]);
    }
    double peak = equity.front();
    double drawdown = 0.0;
    for (std::size_t i = 0; i < equity.size(); ++i) {
        peak = std::max(peak, equity[i]);
        if (peak > 0.0) {
            drawdown = std::min(drawdown, equity[i] / peak - 1.0);
        }
    }
    result.max_drawdown = drawdown;
    if (std::isfinite(result.annualized_return) && drawdown < 0.0) {
        result.calmar = result.annualized_return / std::fabs(drawdown);
    } else {
        result.calmar = std::numeric_limits<double>::quiet_NaN();
    }

    const TailRisk tail =
        historical_var_cvar(finite.data(), finite.size(), confidence);
    result.var = tail.var;
    result.cvar = tail.cvar;
    result.sharpe =
        sharpe(finite.data(), finite.size(), periods_per_year, risk_free);
    result.sortino =
        sortino(finite.data(), finite.size(), periods_per_year, risk_free);
    std::size_t hits = 0;
    for (double value : finite) {
        if (value > 0.0) ++hits;
    }
    result.hit_rate =
        static_cast<double>(hits) / static_cast<double>(finite.size());
    return result;
}

}  // namespace rflux::qflux
